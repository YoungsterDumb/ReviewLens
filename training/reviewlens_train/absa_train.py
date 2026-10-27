"""Fine-tune ATE (DistilBERT, BIO) and ATSC (multilingual ALBERT, sentence pairs) on EN + VI."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from . import SEED
from .data import to_bio, to_pairs
from app.core.bio import align_word_labels, build_label_list
from app.models.text_models import SENTIMENTS

ATE_BASE = "distilbert-base-multilingual-cased"
ATSC_BASE = "cservan/multilingual-albert-base-cased-32k"


def _args(out_dir: Path, epochs: int, lr: float, batch_size: int, metric: str, **extra):
    import torch
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(out_dir), eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model=metric, greater_is_better=True, save_total_limit=1, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs, weight_decay=0.01, warmup_ratio=0.1, logging_steps=50,
        fp16=torch.cuda.is_available(), report_to="none", seed=SEED, **extra,
    )


# --- Aspect term extraction ---------------------------------------------------

def encode_bio(tokenizer, records: list[dict], label2id: dict[str, int], max_length: int = 320):
    from datasets import Dataset

    def enc(batch):
        out = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
        out["labels"] = [align_word_labels(out.word_ids(i), [label2id[t] for t in tags])
                         for i, tags in enumerate(batch["tags"])]
        return out

    return Dataset.from_list(records).map(enc, batched=True, remove_columns=["tokens", "tags", "lang"])


def _decode(pred_ids: np.ndarray, label_ids: np.ndarray, id2label: dict[int, str]):
    preds, golds = [], []
    for p_row, l_row in zip(pred_ids, label_ids):
        keep = l_row != -100
        preds.append([id2label[int(i)] for i in p_row[keep]])
        golds.append([id2label[int(i)] for i in l_row[keep]])
    return preds, golds


def train_ate(splits: dict[str, list[dict]], out_dir: Path, model_name: str = ATE_BASE, epochs: int = 5,
              lr: float = 5e-5, batch_size: int = 16, max_steps: int = -1):
    from seqeval.metrics import f1_score
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              Trainer)

    bio = {k: to_bio(v) for k, v in splits.items()}
    types = sorted({t[2:] for r in bio["train"] for t in r["tags"] if t != "O"})
    labels = build_label_list(types)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = dict(enumerate(labels))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels),
                                                            id2label=id2label, label2id=label2id)
    ds = {k: encode_bio(tok, v, label2id) for k, v in bio.items()}

    def metrics(p):
        preds, golds = _decode(np.argmax(p.predictions, axis=-1), p.label_ids, id2label)
        return {"f1": f1_score(golds, preds)}

    trainer = Trainer(model=model, args=_args(out_dir, epochs, lr, batch_size, "f1", max_steps=max_steps),
                      train_dataset=ds["train"], eval_dataset=ds["validation"], processing_class=tok,
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=metrics)
    trainer.train()

    results = {}
    for lang in ("en", "vi"):
        subset = [r for r in bio["test"] if r["lang"] == lang]
        if subset:
            p = trainer.predict(encode_bio(tok, subset, label2id))
            preds, golds = _decode(np.argmax(p.predictions, axis=-1), p.label_ids, id2label)
            results[lang] = round(float(f1_score(golds, preds)), 4)
    p = trainer.predict(ds["test"])
    preds, golds = _decode(np.argmax(p.predictions, axis=-1), p.label_ids, id2label)
    metrics_out = {"ate_f1": results.get("en"), "ate_f1_vi": results.get("vi"),
                   "ate_f1_all": round(float(f1_score(golds, preds)), 4)}
    return trainer, tok, metrics_out


# --- Aspect term sentiment classification ---------------------------------------

def encode_pairs(tokenizer, pairs: list[dict], max_length: int = 256):
    from datasets import Dataset

    label2id = {s: i for i, s in enumerate(SENTIMENTS)}

    def enc(batch):
        out = tokenizer(batch["sentence"], batch["aspect"], truncation="only_first", max_length=max_length)
        out["labels"] = [label2id[l] for l in batch["label"]]
        return out

    return Dataset.from_list(pairs).map(enc, batched=True, remove_columns=["sentence", "aspect", "label", "lang"])


def class_weights(pairs: list[dict]) -> list[float]:
    counts = Counter(p["label"] for p in pairs)
    total = sum(counts.values())
    w = [total / (len(SENTIMENTS) * max(counts.get(s, 1), 1)) for s in SENTIMENTS]
    return [round(x, 4) for x in w]


def train_atsc(splits: dict[str, list[dict]], out_dir: Path, model_name: str = ATSC_BASE, epochs: int = 4,
               lr: float = 3e-5, batch_size: int = 32, max_steps: int = -1):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

    pairs = {k: to_pairs(v) for k, v in splits.items()}
    weights = torch.tensor(class_weights(pairs["train"]), dtype=torch.float)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=dict(enumerate(SENTIMENTS)), label2id={s: i for i, s in enumerate(SENTIMENTS)})
    ds = {k: encode_pairs(tok, v) for k, v in pairs.items()}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"), "accuracy": accuracy_score(p.label_ids, pred)}

    trainer = WeightedTrainer(model=model, args=_args(out_dir, epochs, lr, batch_size, "macro_f1", max_steps=max_steps),
                              train_dataset=ds["train"], eval_dataset=ds["validation"], processing_class=tok,
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()

    out = {}
    for lang in ("en", "vi"):
        subset = [p for p in pairs["test"] if p["lang"] == lang]
        if subset:
            m = trainer.evaluate(encode_pairs(tok, subset), metric_key_prefix=lang)
            out[f"atsc_macro_f1_{lang}"] = round(float(m[f"{lang}_macro_f1"]), 4)
            out[f"atsc_accuracy_{lang}"] = round(float(m[f"{lang}_accuracy"]), 4)
    out["atsc_macro_f1"] = out.get("atsc_macro_f1_en")
    out["atsc_accuracy"] = out.get("atsc_accuracy_en")
    return trainer, tok, out, pairs
