"""Авторитетный отчёт классификации из полного прогона с сигналами.

Применяет ТЕКУЩЕЕ отгруженное правило (gate-правило decide()) к
reports/validation_full/predictions.csv, где для каждого снимка сохранены
talc_pct, talc_vote_prob, fine_prob. Симуляция правила побитово совпадает с
пайплайном (проверено optimize_rule.py: agreement 1.0000), поэтому отчёт
эквивалентен end-to-end прогону с новым правилом, но без повторного тяжёлого
инференса.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["рядовая", "труднообогатимая", "оталькованная"]

# отгруженные пороги (shlifscan/config.py + classify.decide)
TALC_THR = 10.0
FRAC_VOTE_GATE = 0.5
VOTE_THR = 0.6
FINE_THR = 0.5


def apply_rule(df: pd.DataFrame) -> np.ndarray:
    t = df["talc_pct"].to_numpy()
    v = df["talc_vote_prob"].to_numpy()
    f = df["fine_prob"].to_numpy()
    pred = np.where(f >= FINE_THR, ORDER[1], ORDER[0]).astype(object)
    pred[v > VOTE_THR] = ORDER[2]                       # ветка (1б) голос
    pred[(t > TALC_THR) & (v > FRAC_VOTE_GATE)] = ORDER[2]  # ветка (1) доля+гейт
    return pred


def block(y, p, scope) -> str:
    f1 = f1_score(y, p, average="macro", labels=ORDER, zero_division=0)
    acc = (y == p).mean()
    cm = confusion_matrix(y, p, labels=ORDER)
    out = [f"== {scope}: n={len(y)} acc={acc:.3f} macroF1={f1:.3f}",
           "   строки=истина, столбцы=прогноз " + "/".join(ORDER)]
    out += ["   " + " ".join(f"{v:5d}" for v in row) for row in cm]
    out.append(classification_report(y, p, labels=ORDER, zero_division=0))
    return "\n".join(out)


def main() -> None:
    src = ROOT / "reports/validation_full/predictions.csv"
    df = pd.read_csv(src)
    df = df[df.pred != "error"].reset_index(drop=True)
    p_new = apply_rule(df)
    y = df["label"].to_numpy()

    lines = ["ОТЧЁТ: gate-правило (тальк>10% И голос>0.5, либо голос>0.6), "
             "модель талька v5", ""]
    lines.append(block(y, p_new, "ALL"))
    for part in ["ч1", "ч2"]:
        m = df.part == part
        lines.append("\n" + block(y[m.to_numpy()], p_new[m.to_numpy()], part))
    # бинарно
    binm = df.label != "оталькованная"
    yb = df[binm].label.to_numpy()
    pb = pd.Series(p_new)[binm.to_numpy()].replace({"оталькованная": "рядовая"}).to_numpy()
    f1b = f1_score(yb, pb, average="macro", labels=ORDER[:2], zero_division=0)
    lines.append(f"\n== БИНАРНО (тальк исключён): n={len(yb)} "
                 f"acc={(yb==pb).mean():.3f} macroF1={f1b:.3f}")

    text = "\n".join(lines)
    out = ROOT / "reports/validation_full/metrics_report_gated.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("\nсохранено:", out)


if __name__ == "__main__":
    main()
