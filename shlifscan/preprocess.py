"""Предобработка: робастная нормализация, каналы LAB, маска артефактов.

Домен-устойчивость (валидировано на EDA, схема «rules»):
- percentile stretch яркости L (p1..p99.7) на каждый снимок;
- жёлтость считается относительно матрицы: db = b − median(b | тёмные пиксели),
  что убирает разницу баланса белого между «оливковым» (ч1) и «тёмным»
  (ч2/панорамы) доменами съёмки;
- medianBlur по b подавляет хроматические ореолы на границах ярких фаз.

Для панорам константы нормализации считаются ОДИН раз по даунскейлу всей
панорамы и применяются ко всем тайлам (иначе доли фаз плавают между тайлами).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import PreprocessConfig


@dataclass
class NormConstants:
    """Глобальные константы нормализации (на снимок или на панораму)."""

    p_lo: float          # перцентиль L (нижний)
    p_hi: float          # перцентиль L (верхний)
    b_ref: float         # опорная жёлтость тёмной матрицы
    t_split: Optional[float] = None  # порог жёлтости сульфид/серая (см. segment)


@dataclass
class Preprocessed:
    rgb: np.ndarray          # рабочее RGB (возможно, уменьшенное)
    ln: np.ndarray           # нормализованная яркость float32 [0..1]
    db: np.ndarray           # жёлтость относительно матрицы, float32
    valid: np.ndarray        # bool: пиксели, участвующие в анализе
    scale: float             # коэффициент уменьшения относительно исходника
    consts: NormConstants    # использованные константы


def to_lab_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return l, a, b


def detect_scale_bar(rgb: np.ndarray) -> np.ndarray:
    """Маска вшитой шкалы/подписи («300 мкм» на плашке) в нижней полосе снимка."""
    h, w = rgb.shape[:2]
    band_h = max(int(h * 0.12), 40)
    band = rgb[h - band_h:, :]
    g = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    white = (g > 235) & (hsv[..., 1] < 40)
    white = cv2.morphologyEx(
        white.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    mask_band = np.zeros(g.shape, bool)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area > 400 and ww > 40 and area / float(ww * hh) > 0.6:
            pad = 6
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            x1, y1 = min(x + ww + pad, g.shape[1]), min(y + hh + pad, g.shape[0])
            mask_band[y0:y1, x0:x1] = True
    mask = np.zeros((h, w), bool)
    mask[h - band_h:, :] = mask_band
    return mask


def detect_annotation_lines(rgb: np.ndarray) -> np.ndarray:
    """Маска цветных линий ручной разметки (насыщенный синий), чтобы они
    не искажали анализ, если на вход попал уже размеченный снимок."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    blue = (b - np.maximum(r, g)) > 60
    if blue.any():
        blue = cv2.dilate(blue.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return blue


def lab_stats(rgb: np.ndarray) -> dict:
    """Средние и std каналов LAB (для Reinhard-переноса)."""
    l, a, b = to_lab_channels(rgb)
    return {
        "mean": [float(l.mean()), float(a.mean()), float(b.mean())],
        "std": [float(l.std() + 1e-6), float(a.std() + 1e-6), float(b.std() + 1e-6)],
    }


def reinhard_to_reference(rgb: np.ndarray, ref: dict) -> np.ndarray:
    """Reinhard-нормализация: переносит статистики LAB снимка к референсным.

    Референс — статистики обучающей выборки модели талька; выравнивает
    домены съёмки (оливковый/тёмный) перед нейросетевым инференсом.
    """
    l, a, b = to_lab_channels(rgb)
    src = lab_stats(rgb)
    out = []
    for ch, i in ((l, 0), (a, 1), (b, 2)):
        ch = (ch - src["mean"][i]) / src["std"][i]
        out.append(ch * ref["std"][i] + ref["mean"][i])
    l2 = np.clip(out[0] * (255.0 / 100.0), 0, 255)
    a2 = np.clip(out[1] + 128.0, 0, 255)
    b2 = np.clip(out[2] + 128.0, 0, 255)
    lab = np.stack([l2, a2, b2], axis=-1).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def compute_norm_constants(
    rgb: np.ndarray, cfg: PreprocessConfig, valid: Optional[np.ndarray] = None
) -> NormConstants:
    """Считает константы нормализации по снимку (или даунскейлу панорамы)."""
    l, _, b = to_lab_channels(rgb)
    b = cv2.medianBlur(b, 5)
    sel = valid if valid is not None else np.ones(l.shape, bool)
    p_lo, p_hi = np.percentile(l[sel], cfg.norm_percentiles)
    ln = np.clip((l - p_lo) / max(p_hi - p_lo, 1e-3), 0, 1)
    dark = (ln < 0.5) & sel
    b_ref = float(np.median(b[dark])) if dark.sum() > 100 else float(np.median(b[sel]))
    return NormConstants(p_lo=float(p_lo), p_hi=float(p_hi), b_ref=b_ref)


def preprocess(
    rgb: np.ndarray,
    cfg: PreprocessConfig,
    consts: Optional[NormConstants] = None,
    downscale: bool = True,
) -> Preprocessed:
    scale = 1.0
    if downscale:
        long_side = max(rgb.shape[:2])
        if long_side > cfg.analysis_max_side:
            scale = cfg.analysis_max_side / long_side
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    valid = np.ones(rgb.shape[:2], bool)
    if cfg.remove_scale_bar:
        valid &= ~detect_scale_bar(rgb)
    valid &= ~detect_annotation_lines(rgb)

    if consts is None:
        consts = compute_norm_constants(rgb, cfg, valid)

    l, _, b = to_lab_channels(rgb)
    b = cv2.medianBlur(b, 5)
    ln = np.clip(
        (l - consts.p_lo) / max(consts.p_hi - consts.p_lo, 1e-3), 0, 1
    ).astype(np.float32)
    db = (b - consts.b_ref).astype(np.float32)

    return Preprocessed(rgb=rgb, ln=ln, db=db, valid=valid, scale=scale, consts=consts)
