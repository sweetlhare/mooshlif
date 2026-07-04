"""Классификация срастаний сульфидов: обычные vs тонкие.

Геологическая логика: тонкое срастание — сульфидный агрегат, значительно
замещённый нерудной фазой (тонкие ламели/решётка). Обычное — крупный
сплошной сульфид с малым замещением.

Признаки валидированы EDA на 52 снимках обоих доменов (AUC):
- thick_med (2×медиана distance transform сульфида): 0.976
- n_comp_per_kA (компонент на 1000 px сульфидной площади): 0.962
- perim_per_area: 0.970
Правило «thick_med < 8 px (при ширине 2000) ИЛИ n_comp_per_kA > 2.6 →
тонкие» даёт ~0.94 accuracy на уровне снимка.

Здесь то же правило применяется на уровне АГРЕГАТА (группа сульфидных
фрагментов, объединённых морфологическим закрытием), чтобы получить
попиксельную раскраску обычные/тонкие; агрегатные метки затем
агрегируются в доли по снимку.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import IntergrowthConfig


@dataclass
class IntergrowthResult:
    ordinary: np.ndarray     # bool: сульфиды в обычных срастаниях
    fine: np.ndarray         # bool: сульфиды в тонких срастаниях
    aggregates: np.ndarray   # int32: метки агрегатов (0 = фон)
    features: list[dict]     # признаки по агрегатам (объяснимость)


FEATURE_NAMES = [
    "sulf_frac", "thick_med", "thick_mean", "thick_p90",
    "n_comp_per_ka", "perim_per_area", "inclusion_frac",
    "comp_area_med_ka", "comp_area_p90_ka", "solidity_large",
    "ecd_p50", "ecd_p80",
]


def image_level_features(sulfide: np.ndarray, width_scale: float) -> dict:
    """Признаки срастаний по всему снимку (интерпретируемый вектор для
    классификатора и отчёта). Все длины приведены к референсной ширине
    (умножены на width_scale), площади — в тысячах приведённых px (ka).

    - thick_*: толщина сульфидных структур по 2×distance transform;
    - n_comp_per_ka: фрагментированность (компонент на 1000 px сульфида);
    - perim_per_area: удельная длина границ (аналог PSIA);
    - inclusion_frac: замещённость — доля тёмных включений внутри
      морфологического закрытия сульфидной маски;
    - ecd_p50/p80: гранулометрия — перцентили эквивалентного диаметра
      компонент, взвешенные по площади.
    """
    zeros = {k: 0.0 for k in FEATURE_NAMES}
    s_area = int(sulfide.sum())
    if s_area == 0:
        return zeros

    s8 = sulfide.astype(np.uint8)
    dist = cv2.distanceTransform(s8, cv2.DIST_L2, 3)
    dvals = dist[sulfide] * (2.0 * width_scale)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(s8, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    areas_ka = areas * (width_scale ** 2) / 1000.0

    contours, _ = cv2.findContours(s8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)

    # замещённость: тёмные включения внутри закрытых сульфидных агрегатов
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(s8, cv2.MORPH_CLOSE, k)
    c_area = int(closed.sum())
    inclusion_frac = 1.0 - s_area / c_area if c_area > 0 else 0.0

    # solidity крупных компонент (>= 500 px)
    solidity = []
    big = np.argsort(areas)[::-1][:20]
    for i in big:
        if areas[i] < 500:
            break
        comp = (labels == i + 1).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in cnts]))
        ha = cv2.contourArea(hull)
        if ha > 0:
            solidity.append(areas[i] / ha)

    # гранулометрия: area-weighted перцентили ECD
    ecd = np.sqrt(4.0 * areas / np.pi) * width_scale
    order = np.argsort(ecd)
    cum = np.cumsum(areas[order]) / areas.sum()
    ecd_p50 = float(ecd[order][np.searchsorted(cum, 0.5)])
    ecd_p80 = float(ecd[order][np.searchsorted(cum, 0.8)])

    return {
        "sulf_frac": s_area / sulfide.size,
        "thick_med": float(np.median(dvals)),
        "thick_mean": float(dvals.mean()),
        "thick_p90": float(np.percentile(dvals, 90)),
        "n_comp_per_ka": float((n - 1) / max(areas_ka.sum(), 1e-6)),
        "perim_per_area": float(perim / s_area / max(width_scale, 1e-6)),
        "inclusion_frac": float(inclusion_frac),
        "comp_area_med_ka": float(np.median(areas_ka)),
        "comp_area_p90_ka": float(np.percentile(areas_ka, 90)),
        "solidity_large": float(np.mean(solidity)) if solidity else 0.0,
        "ecd_p50": ecd_p50,
        "ecd_p80": ecd_p80,
    }


def granulometry_curve(sulfide: np.ndarray, width_scale: float = 1.0,
                       n_points: int = 30) -> dict:
    """Гранулометрия сульфидов: кумулятивная кривая ECD, взвешенная по площади.

    Возвращает {"ecd_p50", "ecd_p80", "curve": [[ecd_px, cum_frac], ...]}.
    Без физической калибровки ECD выражен в px рабочего масштаба.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        sulfide.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return {"ecd_p50": 0.0, "ecd_p80": 0.0, "curve": []}
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    ecd = np.sqrt(4.0 * areas / np.pi) * width_scale
    order = np.argsort(ecd)
    ecd_sorted = ecd[order]
    cum = np.cumsum(areas[order]) / areas.sum()
    p50 = float(ecd_sorted[np.searchsorted(cum, 0.5)])
    p80 = float(ecd_sorted[np.searchsorted(cum, 0.8)])
    idx = np.unique(np.linspace(0, len(ecd_sorted) - 1, n_points).astype(int))
    curve = [[round(float(ecd_sorted[i]), 1), round(float(cum[i]), 4)] for i in idx]
    return {"ecd_p50": round(p50, 1), "ecd_p80": round(p80, 1), "curve": curve}


def classify_intergrowths(
    sulfide: np.ndarray, cfg: IntergrowthConfig, work_width: int | None = None
) -> IntergrowthResult:
    h, w = sulfide.shape
    width_scale = cfg.ref_width / float(work_width or w)

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * cfg.aggregate_closing_px + 1, 2 * cfg.aggregate_closing_px + 1),
    )
    closed = cv2.morphologyEx(sulfide.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n_agg, agg_labels = cv2.connectedComponents(closed, connectivity=8)

    ordinary = np.zeros_like(sulfide, bool)
    fine = np.zeros_like(sulfide, bool)
    feats: list[dict] = []

    if n_agg <= 1:
        return IntergrowthResult(ordinary, fine, agg_labels.astype(np.int32), feats)

    dist = cv2.distanceTransform(sulfide.astype(np.uint8), cv2.DIST_L2, 3)
    n_sub, sub_labels = cv2.connectedComponents(sulfide.astype(np.uint8), connectivity=8)

    for lbl in range(1, n_agg):
        agg = agg_labels == lbl
        s_in = sulfide & agg
        s_area = int(s_in.sum())
        if s_area == 0:
            continue
        thick_med = float(2.0 * np.median(dist[s_in])) * width_scale
        n_frag = len(np.unique(sub_labels[s_in]))
        n_comp_per_ka = n_frag / (s_area / 1000.0)

        if s_area < cfg.min_aggregate_px:
            # микрозёрна: единичное мелкое вкрапление не образует срастания;
            # относим по толщине (масштабированной)
            is_fine = thick_med < cfg.thickness_thr_px
        else:
            is_fine = (thick_med < cfg.thickness_thr_px) or (
                n_comp_per_ka > cfg.n_comp_per_ka_thr
            )

        feats.append({
            "label": int(lbl),
            "sulfide_px": s_area,
            "thick_med": round(thick_med, 2),
            "n_comp_per_ka": round(float(n_comp_per_ka), 3),
            "type": "fine" if is_fine else "ordinary",
        })
        if is_fine:
            fine |= s_in
        else:
            ordinary |= s_in

    return IntergrowthResult(ordinary, fine, agg_labels.astype(np.int32), feats)
