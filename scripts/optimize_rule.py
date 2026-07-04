"""Групповая CV-оптимизация составного правила классификации.

По готовому прогону (reports/validation_full/predictions.csv) со всеми
непрерывными сигналами (talc_pct калиброванный, talc_vote_prob, fine_prob)
офлайн подбирает пороги правила:

    (1) talc_pct  > talc_thr           → оталькованная
    (2) vote_prob > vote_thr           → оталькованная
    (3) fine_prob >= fine_thr → труднообогатимая, иначе рядовая

Честность: 5-fold GroupKFold. Группа = физический аншлиф (ч1 — префикс имени,
ч2 — файл) с объединением точных дубликатов по MD5. В каждом фолде пороги
выбираются на train-долях, оцениваются на held-out — это несмещённая оценка
прироста macro-F1. Отдельно печатаются пороги-победители на всех данных.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["рядовая", "труднообогатимая", "оталькованная"]
T_ORD, T_FIN, T_TALC = ORDER


def md5(path: str) -> str:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return path


def build_groups(df: pd.DataFrame) -> np.ndarray:
    """Группа = префикс образца (ч1) / файл (ч2), дубликаты по MD5 слиты."""
    keys = []
    for _, r in df.iterrows():
        stem = Path(r["path"]).stem
        if r["part"] == "ч1":
            keys.append("ч1:" + stem.split()[0])   # номер образца до первого пробела
        else:
            keys.append("ч2:" + stem)
    keys = np.array(keys, dtype=object)

    # union-find для слияния точных дубликатов
    parent = {k: k for k in set(keys)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_hash: dict[str, str] = {}
    for k, p in zip(keys, df["path"]):
        h = md5(p)
        if h in by_hash:
            union(k, by_hash[h])
        else:
            by_hash[h] = k
    return np.array([find(k) for k in keys], dtype=object)


def simulate(sub: pd.DataFrame, talc_thr: float, vote_thr: float,
             fine_thr: float) -> np.ndarray:
    talc = sub["talc_pct"].to_numpy()
    vote = sub["talc_vote_prob"].to_numpy()
    fine = sub["fine_prob"].to_numpy()
    pred = np.where(fine >= fine_thr, T_FIN, T_ORD).astype(object)
    pred[vote > vote_thr] = T_TALC           # ветка (2) поверх срастаний
    pred[talc > talc_thr] = T_TALC           # ветка (1) — высший приоритет
    return pred


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=ORDER, zero_division=0)


# сетки порогов
TALC_GRID = [10, 12, 14, 16, 18, 20, 21, 22, 23, 25, 30, 40, 101]
VOTE_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
FINE_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]


def best_on(sub: pd.DataFrame):
    y = sub["label"].to_numpy()
    best, best_thr = -1.0, (10, 0.60, 0.50)
    for t in TALC_GRID:
        for v in VOTE_GRID:
            for fnt in FINE_GRID:
                s = macro_f1(y, simulate(sub, t, v, fnt))
                if s > best:
                    best, best_thr = s, (t, v, fnt)
    return best_thr, best


def main() -> None:
    df = pd.read_csv(ROOT / "reports/validation_full/predictions.csv")
    df = df[df.pred != "error"].reset_index(drop=True)
    y = df["label"].to_numpy()

    base = (10, 0.60, 0.50)
    base_pred = simulate(df, *base)
    # сверка симуляции с реальным pred пайплайна
    agree = (base_pred == df["pred"].to_numpy()).mean()
    print(f"сверка simulate vs пайплайн pred: совпадение {agree:.4f} "
          f"(macroF1 базового правила {macro_f1(y, base_pred):.4f})")

    groups = build_groups(df)
    print(f"групп: {len(set(groups))} на {len(df)} снимков")

    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=5)
    base_scores, opt_scores, chosen = [], [], []
    for tr, va in gkf.split(df, y, groups):
        tr_df, va_df = df.iloc[tr], df.iloc[va]
        thr, _ = best_on(tr_df)          # подбор ТОЛЬКО на train-фолдах
        chosen.append(thr)
        yv = va_df["label"].to_numpy()
        base_scores.append(macro_f1(yv, simulate(va_df, *base)))
        opt_scores.append(macro_f1(yv, simulate(va_df, *thr)))

    print("\n=== честная GroupKFold-оценка (val-фолды) ===")
    print(f"базовое правило  macroF1: {np.mean(base_scores):.4f} ± {np.std(base_scores):.4f}")
    print(f"оптим. правило   macroF1: {np.mean(opt_scores):.4f} ± {np.std(opt_scores):.4f}")
    print(f"прирост: {np.mean(opt_scores)-np.mean(base_scores):+.4f}")
    print("выбранные пороги по фолдам (talc%, vote, fine):")
    for c in chosen:
        print("   ", c)

    # финальные пороги на всех данных (для продакшна) + метрики
    full_thr, full_f1 = best_on(df)
    print(f"\n=== пороги-победители на всех данных: talc>{full_thr[0]}%, "
          f"vote>{full_thr[1]}, fine>={full_thr[2]} ===")
    from sklearn.metrics import classification_report, confusion_matrix
    for tag, thr in [("БАЗА (10, .60, .50)", base), ("ОПТИМ", full_thr)]:
        p = simulate(df, *thr)
        print(f"\n--- {tag} --- macroF1={macro_f1(y,p):.4f} acc={(y==p).mean():.4f}")
        print("   " + "/".join(ORDER))
        print("\n".join("   " + " ".join(f"{v:4d}" for v in row)
                        for row in confusion_matrix(y, p, labels=ORDER)))
        print(classification_report(y, p, labels=ORDER, zero_division=0, digits=3))


if __name__ == "__main__":
    main()
