"""Conditional VAE shared by training (Keras/TensorFlow) and serving (Keras/torch).

Condition c = [one-hot(category) ; p+ v+ + p0 v0 + p- v-], where v are Word2Vec
anchor vectors for positive / neutral / negative words. Training pairs each
image with a random sentiment mix and a matching brightness, so the decoder
learns: more negative -> darker, duller product image.
"""
from __future__ import annotations

import numpy as np

CATEGORIES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CATEGORIES_VI = [
    "Áo thun", "Quần dài", "Áo len", "Váy", "Áo khoác",
    "Dép sandal", "Áo sơ mi", "Giày thể thao", "Túi xách", "Giày cổ cao",
]
SENTIMENTS = ("negative", "neutral", "positive")
IMG = 28
LATENT_DIM = 16


def sentiment_score(probs: dict[str, float]) -> float:
    """Scalar in [-1, 1]: P(positive) - P(negative)."""
    return float(probs.get("positive", 0.0) - probs.get("negative", 0.0))


def brightness_gain(score: float) -> float:
    """Image gain for a sentiment score: -1 -> 0.45 (dark), +1 -> 1.0 (original)."""
    return 0.45 + 0.55 * (float(np.clip(score, -1, 1)) + 1.0) / 2.0


def condition_vector(category: int, probs: dict[str, float], anchors: dict[str, np.ndarray]) -> np.ndarray:
    onehot = np.zeros(len(CATEGORIES), dtype=np.float32)
    onehot[int(category)] = 1.0
    total = sum(max(0.0, float(probs.get(k, 0.0))) for k in SENTIMENTS) or 1.0
    mix = sum((max(0.0, float(probs.get(k, 0.0))) / total) * np.asarray(anchors[k], dtype=np.float32) for k in SENTIMENTS)
    return np.concatenate([onehot, mix.astype(np.float32)])


def random_sentiment_mix(rng: np.random.Generator, n: int) -> np.ndarray:
    """n x 3 Dirichlet mixes in SENTIMENTS order (negative, neutral, positive)."""
    return rng.dirichlet(alpha=[0.6, 0.6, 0.6], size=n).astype(np.float32)


def build_models(cond_dim: int, latent_dim: int = LATENT_DIM, beta: float = 0.5):
    """Return (cvae, encoder, decoder). Keras 3, backend-agnostic layers only."""
    import keras
    from keras import layers, ops

    @keras.saving.register_keras_serializable(package="reviewlens")
    class Sampling(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed = keras.random.SeedGenerator(1337)

        def call(self, inputs):
            mean, logvar = inputs
            eps = keras.random.normal(ops.shape(mean), seed=self.seed)
            return mean + ops.exp(0.5 * logvar) * eps

    @keras.saving.register_keras_serializable(package="reviewlens")
    class KLLoss(layers.Layer):
        """Adds beta * KL / pixels, so it is on the same scale as per-pixel MSE."""

        def __init__(self, beta: float = 0.5, **kwargs):
            super().__init__(**kwargs)
            self.beta = beta

        def call(self, inputs):
            mean, logvar = inputs
            kl = -0.5 * ops.sum(1 + logvar - ops.square(mean) - ops.exp(logvar), axis=-1)
            self.add_loss(self.beta * ops.mean(kl) / (IMG * IMG))
            return mean

        def get_config(self):
            return {**super().get_config(), "beta": self.beta}

    x_in = keras.Input((IMG, IMG, 1), name="image")
    c_in = keras.Input((cond_dim,), name="condition")
    h = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x_in)
    h = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.Flatten()(h)
    h = layers.Concatenate()([h, c_in])
    h = layers.Dense(256, activation="relu")(h)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(h)
    encoder = keras.Model([x_in, c_in], [z_mean, z_logvar], name="encoder")

    z_in = keras.Input((latent_dim,), name="z")
    c_in2 = keras.Input((cond_dim,), name="condition")
    d = layers.Concatenate()([z_in, c_in2])
    d = layers.Dense(256, activation="relu")(d)
    d = layers.Dense(7 * 7 * 64, activation="relu")(d)
    d = layers.Reshape((7, 7, 64))(d)
    d = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(d)
    d = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(d)
    out = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid", name="image_out")(d)
    decoder = keras.Model([z_in, c_in2], out, name="decoder")

    mean, logvar = encoder([x_in, c_in])
    mean = KLLoss(beta=beta, name="kl")([mean, logvar])
    z = Sampling(name="sampling")([mean, logvar])
    recon = decoder([z, c_in])
    cvae = keras.Model([x_in, c_in], recon, name="cvae")
    return cvae, encoder, decoder
