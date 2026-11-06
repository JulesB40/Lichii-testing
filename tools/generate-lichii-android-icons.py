#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Lichii Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the Android launcher, splash, banner and notification icons from dist/branding.

Usage: python3 -I tools/generate-lichii-android-icons.py  (needs Pillow and numpy)
"""

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
BRANDING = ROOT / "dist" / "branding"
RES = ROOT / "src" / "android" / "app" / "src" / "main" / "res"

# Keep in sync with lichii_launcher_bg_* in res/values/colors.xml.
BG_TOP = (0x4A, 0x14, 0x33)
BG_BOTTOM = (0x1C, 0x08, 0x14)


def clean(path: Path) -> Image.Image:
    """Drop the speckles around the sticker outline and crop to the artwork."""
    arr = np.array(Image.open(path).convert("RGBA"))
    alpha = arr[..., 3]
    mask = alpha > 128
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    sizes = {}
    cur = 0
    for y in range(h):
        for x in np.nonzero(mask[y] & (labels[y] == 0))[0]:
            if labels[y, x]:
                continue
            cur += 1
            q = deque([(y, x)])
            labels[y, x] = cur
            n = 0
            while q:
                cy, cx = q.popleft()
                n += 1
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = cur
                        q.append((ny, nx))
            sizes[cur] = n
    keep = np.isin(labels, [k for k, n in sizes.items() if n >= 3000])
    keep = Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
    arr[..., 3] = (np.array(keep) / 255.0 * alpha).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.crop(img.getbbox())


def gradient(size: tuple[int, int]) -> Image.Image:
    w, h = size
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    top = np.array(BG_TOP, float)[None, None, :]
    bottom = np.array(BG_BOTTOM, float)[None, None, :]
    rows = (top * (1 - t) + bottom * t).astype(np.uint8)
    rgb = np.repeat(rows, w, axis=1)
    return Image.fromarray(np.dstack([rgb, np.full((h, w), 255, np.uint8)]), "RGBA")


def place(canvas: Image.Image, art: Image.Image, fraction: float) -> Image.Image:
    """Center art on canvas, scaled to fit fraction of the canvas."""
    cw, ch = canvas.size
    scale = fraction * min(cw / art.width, ch / art.height)
    art = art.resize((round(art.width * scale), round(art.height * scale)), Image.LANCZOS)
    canvas.alpha_composite(art, ((cw - art.width) // 2, (ch - art.height) // 2))
    return canvas


def rounded(img: Image.Image, radius_fraction: float) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    r = round(min(img.size) * radius_fraction)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, img.width - 1, img.height - 1), r, fill=255)
    out = img.copy()
    out.putalpha(mask)
    return out


def main() -> None:
    icon = clean(BRANDING / "lichii_icon.png")
    logo = clean(BRANDING / "lichii_icon_with_logo.png")

    # Adaptive icon foreground: 108dp canvas, the artwork inside the 66dp safe zone.
    place(Image.new("RGBA", (432, 432)), icon, 0.58).save(RES / "drawable" / "ic_launcher_foreground.png")

    # Themed (monochrome) icon for Android 13+.
    silhouette = Image.new("RGBA", icon.size, (255, 255, 255, 0))
    silhouette.putalpha(icon.getchannel("A"))
    place(Image.new("RGBA", (432, 432)), silhouette, 0.58).save(RES / "drawable" / "ic_launcher_monochrome.png")

    # Legacy launcher icons (48dp).
    for density, px in {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}.items():
        big = rounded(place(gradient((512, 512)), icon, 0.78), 0.22)
        big.resize((px, px), Image.LANCZOS).save(RES / f"mipmap-{density}" / "ic_launcher.png")

    # Splash: the system masks the icon to a circle of 2/3 of the canvas.
    place(Image.new("RGBA", (512, 512)), icon, 0.52).save(RES / "drawable" / "ic_yuzu_splash.png")

    # About page and document provider.
    place(Image.new("RGBA", (512, 512)), icon, 0.96).save(RES / "drawable" / "ic_yuzu.png")

    # Wordmark for the setup and home screens.
    place(Image.new("RGBA", (1200, 545)), logo, 1.0).save(RES / "drawable" / "lichii_logo.png")

    # Android TV banner.
    place(gradient((640, 360)), logo, 0.82).convert("RGB").resize((320, 180), Image.LANCZOS).save(
        RES / "drawable-xhdpi" / "tv_banner.png")

    # Notification icon (24dp).
    for density, px in {"hdpi": 36, "xhdpi": 48, "xxhdpi": 72}.items():
        canvas = place(Image.new("RGBA", (px * 4, px * 4)), silhouette, 0.9)
        canvas.resize((px, px), Image.LANCZOS).save(RES / f"drawable-{density}" / "ic_stat_notification_logo.png")


if __name__ == "__main__":
    main()
