"""Генерация Deep Zoom (DZI) пирамид для OpenSeadragon.

Основной путь — pyvips (потоковый, быстрый). Фолбэк — PIL (медленнее,
но без системных зависимостей).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

TILE = 512
OVERLAP = 1


def _dzi_xml(width: int, height: int, fmt: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
        f'Format="{fmt}" Overlap="{OVERLAP}" TileSize="{TILE}">'
        f'<Size Width="{width}" Height="{height}"/></Image>'
    )


def make_dzi_from_file(src: str | Path, out_base: str | Path, fmt: str = "jpg") -> None:
    """DZI из файла изображения (исходник-панорама)."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyvips

        im = pyvips.Image.new_from_file(str(src), access="sequential")
        im.dzsave(
            str(out_base), tile_size=TILE, overlap=OVERLAP,
            suffix=f".{fmt}[Q=90]" if fmt == "jpg" else f".{fmt}",
        )
        return
    except Exception:
        pass
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as im:
        _pil_dzi(im.convert("RGB"), out_base, fmt)


def make_dzi_from_array(arr: np.ndarray, out_base: str | Path, fmt: str = "png") -> None:
    """DZI из numpy-массива (RGBA-маска фаз)."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyvips

        h, w = arr.shape[:2]
        bands = arr.shape[2] if arr.ndim == 3 else 1
        im = pyvips.Image.new_from_memory(
            np.ascontiguousarray(arr).tobytes(), w, h, bands, "uchar"
        )
        im.dzsave(str(out_base), tile_size=TILE, overlap=OVERLAP, suffix=f".{fmt}")
        return
    except Exception:
        pass
    from PIL import Image

    _pil_dzi(Image.fromarray(arr), out_base, fmt)


def _pil_dzi(im, out_base: Path, fmt: str) -> None:
    """Фолбэк-генератор пирамиды на PIL."""
    from PIL import Image

    w, h = im.size
    max_level = max(int(math.ceil(math.log2(max(w, h)))), 0)
    (out_base.parent / f"{out_base.name}.dzi").write_text(_dzi_xml(w, h, fmt))
    files_dir = out_base.parent / f"{out_base.name}_files"

    level_im = im
    for level in range(max_level, -1, -1):
        lw, lh = level_im.size
        ld = files_dir / str(level)
        ld.mkdir(parents=True, exist_ok=True)
        cols = math.ceil(lw / TILE)
        rows = math.ceil(lh / TILE)
        for r in range(rows):
            for c in range(cols):
                x0 = max(c * TILE - OVERLAP, 0)
                y0 = max(r * TILE - OVERLAP, 0)
                x1 = min((c + 1) * TILE + OVERLAP, lw)
                y1 = min((r + 1) * TILE + OVERLAP, lh)
                tile = level_im.crop((x0, y0, x1, y1))
                if fmt == "jpg":
                    tile.convert("RGB").save(ld / f"{c}_{r}.jpg", quality=90)
                else:
                    tile.save(ld / f"{c}_{r}.{fmt}")
        if level > 0:
            level_im = level_im.resize(
                (max(lw // 2, 1), max(lh // 2, 1)), Image.BILINEAR
            )
