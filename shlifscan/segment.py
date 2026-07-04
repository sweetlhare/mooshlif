"""Сегментация фаз: сульфиды / серая нерудная фаза / тёмная матрица.

Схема «rules» (победитель EDA-сравнения с multi-Otsu и KMeans):
пиксель — сульфид, если он ярок (Ln > 0.55) и жёлт относительно матрицы
(db ≥ t_split); серая фаза — ярче матрицы, но нейтральна по цвету.
Порог жёлтости t_split подбирается per-image методом Otsu по db среди
ярких пикселей с защитами от вырожденных случаев — это даёт
кросс-доменную устойчивость без ручной подстройки (t≈10–13 в оливковом
домене, t≈3–7 в тёмном).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SegmentConfig
from .preprocess import Preprocessed


T_SULF_L = 0.55   # сульфид: достаточно ярко (по нормализованной L)
T_GRAY_L = 0.30   # серая фаза: заметно светлее матрицы


@dataclass
class PhaseSegmentation:
    sulfide: np.ndarray      # bool
    gray: np.ndarray         # bool
    matrix: np.ndarray       # bool
    confidence: np.ndarray   # float32 [0..1]
    t_split: float           # использованный порог жёлтости


def adaptive_b_split(ln: np.ndarray, db: np.ndarray, valid: np.ndarray) -> float:
    """Per-image порог жёлтости (сульфид vs серая фаза) с защитами."""
    from skimage.filters import threshold_otsu

    bright = (ln > T_GRAY_L) & valid
    if bright.sum() < 500:
        return 6.0
    v = np.clip(db[bright], -10, 30)
    try:
        t = float(threshold_otsu(v))
    except ValueError:
        return 6.0
    lm = v[v < t].mean() if (v < t).any() else -10.0
    um = v[v >= t].mean() if (v >= t).any() else 30.0
    if um < 6:   # нет жёлтой (сульфидной) моды — всё яркое = серая фаза
        return 99.0
    if lm > 8:   # нет нейтральной моды — всё яркое = сульфид
        return 3.0
    return float(np.clip(t, 3, 14))


def _open_close(mask: np.ndarray, open_px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3)
    return m.astype(bool)


def segment_phases(
    pre: Preprocessed, cfg: SegmentConfig, t_split: float | None = None
) -> PhaseSegmentation:
    ln, db, valid = pre.ln, pre.db, pre.valid

    if t_split is None:
        t_split = pre.consts.t_split
    if t_split is None:
        t_split = adaptive_b_split(ln, db, valid)

    sulfide = (ln > T_SULF_L) & (db >= t_split)
    sulfide |= (ln > 0.70) & (db > 10)          # строгий абсолютный fallback
    gray = (ln > T_GRAY_L) & (db < t_split) & ~sulfide

    # открытие давит шум и хроматические ореолы, закрытие сшивает трещинки
    sulfide = _open_close(sulfide, cfg.sulf_open_px) & valid
    gray = _open_close(gray, cfg.gray_open_px) & valid & ~sulfide
    matrix = valid & ~sulfide & ~gray

    # уверенность: расстояние до решающих порогов (яркостного и цветового)
    d_l = np.minimum(np.abs(ln - T_SULF_L), np.abs(ln - T_GRAY_L)) / 0.15
    d_b = np.abs(db - t_split) / 6.0
    confidence = np.clip(np.minimum(1.0, 0.4 + 0.6 * np.minimum(d_l, d_b)), 0, 1)
    confidence = confidence.astype(np.float32)
    confidence[~valid] = 0.0

    return PhaseSegmentation(
        sulfide=sulfide, gray=gray, matrix=matrix,
        confidence=confidence, t_split=float(t_split),
    )
