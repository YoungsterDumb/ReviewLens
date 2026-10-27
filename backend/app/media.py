"""Image helpers: safe decoding, PNG data URIs, LIME heat-map overlays."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 1024


class BadImage(ValueError):
    pass


def load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise BadImage("The file is not a readable image (use JPG, PNG or WebP).") from exc
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def gray_to_data_uri(arr: np.ndarray, scale: int = 4) -> str:
    """28x28 float [0,1] -> upscaled grayscale PNG (nearest neighbour keeps the pixels crisp)."""
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(a, mode="L")
    if scale > 1:
        img = img.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    return to_data_uri(img)


def heatmap_overlay(image: np.ndarray, segments: np.ndarray, weights: dict[int, float]) -> Image.Image:
    """Tint superpixels: coral where they push toward the label, slate where they push against."""
    base = image.astype(float)
    if base.max() <= 1.0:
        base = base * 255.0
    out = base * 0.55 + 255 * 0.45 * np.ones_like(base)  # soften the photo under the overlay
    vmax = max((abs(w) for w in weights.values()), default=0.0) or 1.0
    coral, slate = np.array([232, 106, 87.0]), np.array([74, 93, 138.0])
    for seg, w in weights.items():
        mask = segments == seg
        alpha = 0.75 * min(1.0, abs(w) / vmax)
        color = coral if w > 0 else slate
        out[mask] = out[mask] * (1 - alpha) + color * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
