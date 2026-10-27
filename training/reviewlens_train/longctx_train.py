"""Long-context toxicity: XLM-R Longformer (4,096 tokens) fine-tuned on Jigsaw Toxic Comment."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import SEED

LONG_BASE = "AshtonIsNotHere/xlm-roberta-long-base-4096"
JIGSAW = "thesofakillers/jigsaw-toxic-comment-classification-challenge"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_jigsaw(n_train: int = 16_000, n_val: int = 2_000, n_test: int = 3_000, seed: int = SEED):
    """Pool Jigsaw's labelled train.csv, split 70/15/15 by toxicity, then subsample for a T4 budget.

    Every comment longer than 512 tokens in the test split is kept, so long-context coverage is measured.
    """
    import pandas as pd
    from datasets import load_dataset

    df = load_dataset(JIGSAW, split="train").to_pandas()
    df = df.drop_duplicates("comment_text")
    df["any"] = (df[LABELS].sum(axis=1) > 0).astype(int)
    rng = np.random.default_rng(seed)
    parts = {"train": [], "validation": [], "test": []}
    for _, g in df.groupby("any"):
        idx = rng.permutation(len(g))
        a, b = int(len(g) * 0.7), int(len(g) * 0.85)
        parts["train"].append(g.iloc[idx[:a]])
        parts["validation"].append(g.iloc[idx[a:b]])
        parts["test"].append(g.iloc[idx[b:]])
    splits = {k: pd.concat(v) for k, v in parts.items()}

    def sample(frame, n, toxic_share=0.4):
        tox, clean = frame[frame["any"] == 1], frame[frame["any"] == 0]
        nt = min(len(tox), int(n * toxic_share))
        return pd.concat([tox.sample(nt, random_state=seed), clean.sample(min(len(clean), n - nt), random_state=seed)])

    long_mask = splits["test"]["comment_text"].str.len() > 2_000  # ~>512 tokens; exact count below
    test = pd.concat([sample(splits["test"], n_test), splits["test"][long_mask]]).drop_duplicates("id")
    return {"train": sample(splits["train"], n_train), "validation": sample(splits["validation"], n_val), "test": test}


def train_longformer(frames: dict, out_dir: Path, model_name: str = LONG_BASE, epochs: int = 1, lr: float = 2e-5,
                     train_max_length: int = 1024, max_steps: int = -1):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})

    def to_ds(frame, max_length):
        ds = Dataset.from_pandas(frame[["comment_text"] + LABELS].reset_index(drop=True))

        def enc(batch):
            out = tok(batch["comment_text"], truncation=True, max_length=max_length)
            out["labels"] = [[float(batch[l][i]) for l in LABELS] for i in range(len(batch["comment_text"]))]
            return out

        return ds.map(enc, batched=True, remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir=str(out_dir), eval_strategy="no", save_strategy="no", learning_rate=lr,
        per_device_train_batch_size=4, gradient_accumulation_steps=4, num_train_epochs=epochs, max_steps=max_steps,
        warmup_ratio=0.06, weight_decay=0.01, logging_steps=50, fp16=torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available(), group_by_length=True, report_to="none", seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=to_ds(frames["train"], train_max_length),
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    return trainer, tok


def evaluate_longformer(model, tok, frame, max_length: int = 4096, batch_size: int = 4) -> dict:
    """Macro ROC-AUC on the test split, reading each comment whole (up to 4,096 tokens)."""
    import torch
    from sklearn.metrics import roc_auc_score

    device = next(model.parameters()).device
    texts = frame["comment_text"].tolist()
    lengths = [len(tok(t)["input_ids"]) for t in texts]
    order = np.argsort(lengths)
    probs = np.zeros((len(texts), len(LABELS)))
    model.eval()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(**enc).logits.float().cpu().numpy()
        probs[idx] = 1 / (1 + np.exp(-logits))
    y = frame[LABELS].to_numpy()
    aucs = [roc_auc_score(y[:, j], probs[:, j]) for j in range(len(LABELS)) if 0 < y[:, j].sum() < len(y)]
    long = [n for n in lengths if n > 512]
    return {
        "long_context_auc": round(float(np.mean(aucs)), 4),
        "long_context_coverage": round(float(np.mean([n <= max_length for n in long])), 4) if long else 1.0,
        "long_context_max_tokens": int(max(lengths)),
        "long_context_long_comments": len(long),
    }
