"""Стереологические оценки погрешности долей фаз.

Методология: принцип Delesse (Aᴀ = Vᵥ, площадная доля = объёмная),
межпольная оценка доверительного интервала по ASTM E562
(CI = t·s/√n по полям измерения), тест избыточной дисперсии MSWD
(Vermeesch 2018) как флаг пространственной неоднородности фазы.

Поле измерения = ячейка сетки, на которую делится фазовая карта
(для панорамы ~ соответствует тайлам обработки).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FractionCI:
    mean_pct: float          # площадно-взвешенная средняя доля, %
    ci_low_pct: float        # нижняя граница 95% CI (полная), %
    ci_high_pct: float       # верхняя граница, %
    spatial_ci_pct: float    # межпольная компонента (E562), п.п.
    model_ci_pct: float      # модельная компонента (MAE калибровки), п.п.
    mswd: float              # избыточная дисперсия (≈1 — только счётный шум)
    heterogeneous: bool      # True → фаза распределена неоднородно
    n_fields: int


def _t95(df: int) -> float:
    """Квантиль Стьюдента t(0.975, df) без scipy-зависимости в рантайме."""
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, max(df, 1)))
    except Exception:
        table = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
                 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09,
                 30: 2.04, 60: 2.00}
        keys = sorted(table)
        for k in keys:
            if df <= k:
                return table[k]
        return 1.96


def fraction_ci(
    phase_mask: np.ndarray,
    valid: np.ndarray,
    model_mae_pct: float = 0.0,
    grid: int = 6,
    calibrate=None,
) -> FractionCI:
    """95% CI доли фазы по межпольной дисперсии (сетка grid×grid полей).

    calibrate — опциональная функция калибровки доли (frac→frac),
    применяется к пополевым долям, чтобы CI жил в калиброванной шкале.
    """
    h, w = phase_mask.shape
    fracs, areas = [], []
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    for i in range(grid):
        for j in range(grid):
            v = valid[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            a = int(v.sum())
            if a < 500:      # поле почти пустое (фон/шкала) — пропускаем
                continue
            m = phase_mask[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            f = float((m & v).sum()) / a
            if calibrate is not None:
                f = float(calibrate(f))
            fracs.append(f)
            areas.append(a)

    if len(fracs) < 2:
        p = float(np.mean(fracs)) * 100 if fracs else 0.0
        half = model_mae_pct
        return FractionCI(round(p, 2), round(max(p - half, 0), 2),
                          round(p + half, 2), 0.0, model_mae_pct,
                          0.0, False, len(fracs))

    fr = np.array(fracs)
    ar = np.array(areas, dtype=np.float64)
    wgt = ar / ar.sum()
    p_bar = float((wgt * fr).sum())

    # эффективное число полей (площади не равны из-за краёв/масок)
    n_eff = float(ar.sum() ** 2 / (ar ** 2).sum())
    s = float(np.sqrt(((fr - p_bar) ** 2 * wgt).sum() * len(fr) / (len(fr) - 1)))
    spatial_half = _t95(int(round(n_eff)) - 1) * s / np.sqrt(n_eff)

    # MSWD: межпольная дисперсия против биномиального счётного шума
    var_count = np.maximum(p_bar * (1 - p_bar) / ar, 1e-12)
    mswd = float(np.mean((fr - p_bar) ** 2 / var_count))

    model_half = model_mae_pct / 100.0
    total_half = float(np.sqrt(spatial_half ** 2 + model_half ** 2))

    return FractionCI(
        mean_pct=round(p_bar * 100, 2),
        ci_low_pct=round(max(p_bar - total_half, 0) * 100, 2),
        ci_high_pct=round(min(p_bar + total_half, 1) * 100, 2),
        spatial_ci_pct=round(spatial_half * 100, 2),
        model_ci_pct=round(model_mae_pct, 2),
        mswd=round(mswd, 1),
        heterogeneous=bool(mswd > 3.0),
        n_fields=len(fracs),
    )


def association_index(
    sulfide: np.ndarray, other: np.ndarray, valid: np.ndarray
) -> float:
    """Индекс ассоциации: доля границы сульфидов, контактирующей с фазой
    `other` (аналог association index геометаллургии, Koch & Lund).

    Контакт сульфид↔тальк — прямой фактор потери селективности флотации.
    """
    import cv2

    s8 = (sulfide & valid).astype(np.uint8)
    if not s8.any():
        return 0.0
    ring = cv2.dilate(s8, np.ones((3, 3), np.uint8)).astype(bool) & ~sulfide & valid
    total = int(ring.sum())
    if total == 0:
        return 0.0
    return round(float((ring & other).sum()) / total, 4)
