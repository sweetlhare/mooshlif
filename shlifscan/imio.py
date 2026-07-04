"""Ввод-вывод изображений: TIFF/PNG/JPEG/BMP любого размера, тайлинг панорам."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # панорамы до гигапикселя

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def imread_rgb(path: str | Path) -> np.ndarray:
    """Читает изображение в RGB uint8. Поддерживает кириллические пути."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile

            arr = tifffile.imread(str(path))
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            if arr.dtype != np.uint8:
                arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return arr[..., :3]
        except Exception:
            pass
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) без полной загрузки в память."""
    with Image.open(path) as im:
        return im.size


@dataclass
class Tile:
    x: int
    y: int
    w: int
    h: int
    # область без перекрытия (для сборки результата без швов)
    core_x: int
    core_y: int
    core_w: int
    core_h: int


def iter_tiles(width: int, height: int, tile: int, overlap: int) -> Iterator[Tile]:
    """Генерирует сетку тайлов с перекрытием.

    core-область каждого тайла покрывает изображение ровно один раз,
    что позволяет собирать маски без двойного учёта и швов.
    """
    step = tile - 2 * overlap
    assert step > 0, "tile должен быть больше 2*overlap"
    ys = list(range(0, max(height - 2 * overlap, 1), step))
    xs = list(range(0, max(width - 2 * overlap, 1), step))
    for cy in ys:
        for cx in xs:
            x0 = max(cx - overlap, 0)
            y0 = max(cy - overlap, 0)
            x1 = min(cx + step + overlap, width)
            y1 = min(cy + step + overlap, height)
            core_x0 = cx
            core_y0 = cy
            core_x1 = min(cx + step, width)
            core_y1 = min(cy + step, height)
            if core_x1 <= core_x0 or core_y1 <= core_y0:
                continue
            yield Tile(
                x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                core_x=core_x0, core_y=core_y0,
                core_w=core_x1 - core_x0, core_h=core_y1 - core_y0,
            )


def read_region(path: str | Path, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Читает регион большого изображения (PIL декодирует JPEG целиком,
    поэтому для многих тайлов эффективнее один раз загрузить снимок;
    эта функция — для точечных обращений)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").crop((x, y, x + w, y + h)))
