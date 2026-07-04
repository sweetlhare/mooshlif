"""A/B морфопризнаков срастаний: старые 12 vs +3 по Дурягиной (сферичность,
вытянутость, межзёрновое расстояние). Бинарная задача рядовая vs
труднообогатимая (тальк исключён), честная 5-fold GroupKFold по группе аншлифа.

Мержит новые признаки из features_v3.csv на чистый features_v2_clean.csv (группы,
дедуп). Сравнивает macro-F1 GBM с одинаковой конфигурацией — важна ОТНОСИТЕЛЬНАЯ
разница.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OLD = ["sulf_frac", "thick_med", "thick_mean", "thick_p90", "n_comp_per_ka",
       "perim_per_area", "inclusion_frac", "comp_area_med_ka", "comp_area_p90_ka",
       "solidity_large", "ecd_p50", "ecd_p80"]
NEW = ["sphericity_large", "elong_mean", "intergrain_med"]


def load() -> pd.DataFrame:
    clean = pd.read_csv(ROOT / "reports/features_v2_clean.csv")
    v3 = pd.read_csv(ROOT / "reports/features_v3.csv")
    # приводим path к basename для устойчивого мержа
    clean["_k"] = clean["path"].map(lambda p: Path(str(p)).name)
    v3["_k"] = v3["path"].map(lambda p: Path(str(p)).name)
    merged = clean.merge(v3[["_k"] + NEW], on="_k", how="left")
    merged = merged.dropna(subset=NEW)
    # бинарная задача: только рядовая / труднообогатимая
    merged = merged[merged.label.isin(["рядовая", "труднообогатимая"])].copy()
    merged["y"] = (merged.label == "труднообогатимая").astype(int)
    return merged


def cv_f1(df: pd.DataFrame, feats: list[str], seed: int = 0) -> tuple[float, float]:
    X = df[feats].to_numpy()
    y = df["y"].to_numpy()
    g = df["group"].astype(str).to_numpy()
    scores = []
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(X, y, g):
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=3,
            l2_regularization=1.0, random_state=seed)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[va])
        scores.append(f1_score(y[va], pred, average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    df = load()
    print(f"снимков (рядовая+трудно): {len(df)} | групп: {df.group.nunique()}")
    print(f"баланс: трудно {df.y.mean():.2%}")
    # усредняем по нескольким сидам (GBM стохастичен, разница мала)
    for tag, feats in [("OLD-12", OLD), ("NEW-15", OLD + NEW)]:
        ms = [cv_f1(df, feats, s) for s in range(5)]
        mean = np.mean([m for m, _ in ms])
        std = np.mean([s for _, s in ms])
        print(f"{tag}: macro-F1 {mean:.4f} ± {std:.4f}  ({len(feats)} признаков)")

    # важность новых признаков (permutation на полной модели)
    from sklearn.inspection import permutation_importance
    X = df[OLD + NEW].to_numpy(); y = df["y"].to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=3, l2_regularization=1.0,
                                         random_state=0).fit(X, y)
    imp = permutation_importance(clf, X, y, n_repeats=10, random_state=0,
                                 scoring="f1_macro")
    order = np.argsort(imp.importances_mean)[::-1]
    print("\nважность признаков (permutation, топ-15):")
    for i in order:
        star = " <-- НОВЫЙ" if (OLD + NEW)[i] in NEW else ""
        print(f"  {(OLD+NEW)[i]:20s} {imp.importances_mean[i]:+.4f}{star}")


if __name__ == "__main__":
    main()
