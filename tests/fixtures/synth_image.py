"""Deterministic synthetic test images for the slow vision tests.

We deliberately avoid committing binary fixtures or relying on a remote
download — both would brittle the slow test on developer machines and HF
Spaces. The shapes are drawn at sizes / colors that OWL-ViT base reliably
picks up at threshold=0.1 when the vocabulary contains "ball" or "cup".
"""

from __future__ import annotations

from PIL import Image, ImageDraw


def make_clevr_like_image(width: int = 480, height: int = 320) -> Image.Image:
    """Return a CLEVR-style scene: solid background with 3 distinct shapes.

    Layout: red sphere (left), blue cube (middle), yellow cylinder (right).
    All shapes are large (>15% of image), high-contrast against the gray
    background, and positioned to exercise spatial relations.
    """
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)

    # Sphere (red disc) — left third
    draw.ellipse([40, 100, 160, 220], fill=(220, 30, 30), outline=(90, 0, 0), width=3)
    # Cube (blue square) — middle third
    draw.rectangle([200, 110, 320, 230], fill=(30, 90, 220), outline=(0, 20, 90), width=3)
    # Cylinder (yellow ellipse vertical) — right third
    draw.ellipse([360, 70, 440, 250], fill=(230, 220, 30), outline=(120, 100, 0), width=3)

    return img
