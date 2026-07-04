"""Визуализация: цветная маска поверх исходника, карта уверенности.

Легенда конкурса: зелёный = обычные срастания, красный = тонкие срастания,
синий = тальк.
"""

from __future__ import annotations

import cv2
import numpy as np

# метки классов в карте фаз (uint8)
PH_BG = 0          # матрица / прочее
PH_ORDINARY = 1    # сульфиды, обычные срастания
PH_FINE = 2        # сульфиды, тонкие срастания
PH_TALC = 3        # тальк
PH_GRAY = 4        # серая нерудная фаза (магнетит) — служебная

COLORS = {
    PH_ORDINARY: (46, 204, 64),    # зелёный
    PH_FINE: (255, 65, 54),        # красный
    PH_TALC: (0, 116, 217),        # синий
}


def overlay(rgb: np.ndarray, phase_map: np.ndarray, alpha: float = 0.45,
            draw_gray: bool = False) -> np.ndarray:
    """Полупрозрачная заливка фаз + контуры для читаемости."""
    out = rgb.copy()
    colors = dict(COLORS)
    if draw_gray:
        colors[PH_GRAY] = (170, 170, 170)
    for label, color in colors.items():
        mask = phase_map == label
        if not mask.any():
            continue
        layer = out.copy()
        layer[mask] = color
        out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 1)
    return out


def confidence_heatmap(rgb: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Карта уверенности: красным подсвечены спорные области (низкая уверенность)."""
    uncert = np.clip(1.0 - conf, 0, 1)
    heat = cv2.applyColorMap((uncert * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(heat, 0.4, rgb, 0.6, 0)


def _find_cyrillic_font(size: int):
    """Ищет системный TTF с поддержкой кириллицы (macOS/Linux)."""
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def legend_strip(width: int = 900, height: int = 46) -> np.ndarray:
    """Горизонтальная легенда для отчётов/контакт-листов (кириллица через PIL)."""
    from PIL import Image, ImageDraw

    strip = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    font = _find_cyrillic_font(height - 24)
    items = [
        ("обычные срастания", COLORS[PH_ORDINARY]),
        ("тонкие срастания", COLORS[PH_FINE]),
        ("тальк", COLORS[PH_TALC]),
    ]
    x = 10
    for text, color in items:
        draw.rectangle([x, 10, x + 26, height - 10], fill=color)
        x += 34
        draw.text((x, 10), text, fill=(30, 30, 30), font=font)
        x += int(draw.textlength(text, font=font)) + 40
    return np.asarray(strip)
