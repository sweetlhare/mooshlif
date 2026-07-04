"""Валидация пайплайна на размеченных папках: прогон всех снимков,
сохранение предсказаний и признаков, расчёт метрик классификации.

Запуск:
    python scripts/validate_classification.py --out reports/validation
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shlifscan.config import PipelineConfig
from shlifscan.imio import SUPPORTED_EXT
from shlifscan.pipeline import analyze_image

DATA = Path(__file__).resolve().parents[1] / "data"

FOLDERS = {
    ("ч1", "рядовая"): DATA / "Фото руд по сортам. ч1/Рядовые руды",
    ("ч1", "труднообогатимая"): DATA / "Фото руд по сортам. ч1/Труднообогатимые руды",
    ("ч1", "оталькованная"): DATA / "Фото руд по сортам. ч1/Оталькованные руды",
    ("ч2", "рядовая"): DATA / "Фото руд по сортам. ч2/рядовые",
    ("ч2", "труднообогатимая"): DATA / "Фото руд по сортам. ч2/тонкие",
    ("ч2", "оталькованная"): DATA / "Фото руд по сортам. ч2/оталькованные",
}


def list_images() -> list[dict]:
    rows = []
    for (part, label), folder in FOLDERS.items():
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXT:
                rows.append({"path": str(f), "part": part, "label": label})
    return rows


def process(item: dict) -> dict:
    cfg = PipelineConfig()
    # снимки ч2 бывают 6240 px — не считать их панорамами
    cfg.pano_min_side = 8000
    try:
        res = analyze_image(item["path"], cfg)
        m = res.metrics
        feats = res.extra.get("image_features", {})
        p = res.verdict.params
        return {
            **item,
            "pred": res.verdict.ore_class,
            "sulfide_pct": round(m.sulfide_total_pct, 3),
            "ordinary_share": round(m.ordinary_share, 2),
            "fine_share": round(m.fine_share, 2),
            "talc_pct": round(m.talc_pct, 3),
            "talc_pct_raw": round(res.extra.get("talc_pct_raw", 0.0), 3),
            "talc_vote_prob": round(float(p.get("talc_vote_prob", -1)), 4),
            "fine_prob": round(float(p.get("fine_prob", -1)), 4),
            "gray_pct": round(m.gray_phase_pct, 3),
            "confidence": round(m.confidence, 3),
            "thick_med": round(feats.get("thick_med", 0), 3),
            "n_comp_per_ka": round(feats.get("n_comp_per_ka", 0), 3),
            "perim_per_area": round(feats.get("perim_per_area", 0), 4),
            "t_split": round(res.extra.get("t_split", 0), 2),
            "elapsed_s": round(res.elapsed_s, 2),
            "error": "",
        }
    except Exception as e:
        traceback.print_exc()
        return {**item, "pred": "error", "error": str(e)}


def compute_report(df: pd.DataFrame, out: Path) -> str:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    ok = df[df.pred != "error"]
    lines = []
    order = ["рядовая", "труднообогатимая", "оталькованная"]
    for scope, sub in [("ALL", ok), ("ч1", ok[ok.part == "ч1"]), ("ч2", ok[ok.part == "ч2"])]:
        if len(sub) == 0:
            continue
        f1m = f1_score(sub.label, sub.pred, average="macro", labels=order, zero_division=0)
        acc = (sub.label == sub.pred).mean()
        cm = confusion_matrix(sub.label, sub.pred, labels=order)
        lines += [
            f"== {scope}: n={len(sub)} acc={acc:.3f} macroF1={f1m:.3f}",
            "   строки=истина, столбцы=прогноз " + "/".join(order),
            "\n".join("   " + " ".join(f"{v:5d}" for v in row) for row in cm),
            classification_report(sub.label, sub.pred, labels=order, zero_division=0),
            "",
        ]
    # отдельно: бинарная задача рядовая-vs-тонкие (исключая talc-класс)
    bin_sub = ok[ok.label != "оталькованная"]
    bin_sub = bin_sub.assign(pred_bin=bin_sub.pred.replace({"оталькованная": "рядовая"}))
    f1b = f1_score(bin_sub.label, bin_sub.pred_bin, average="macro",
                   labels=order[:2], zero_division=0)
    accb = (bin_sub.label == bin_sub.pred_bin).mean()
    lines.append(f"== БИНАРНО (рядовая vs труднообогатимая, тальк исключён): "
                 f"n={len(bin_sub)} acc={accb:.3f} macroF1={f1b:.3f}")
    text = "\n".join(lines)
    (out / "metrics_report.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/validation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = list_images()
    if args.limit:
        items = items[: args.limit]
    print(f"снимков: {len(items)}", flush=True)

    # spawn: fork несовместим с MPS/torch на macOS
    from multiprocessing import get_context

    with get_context("spawn").Pool(args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(process, items, chunksize=4)):
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(items)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    print(compute_report(df, out))
    (out / "config.json").write_text(PipelineConfig().to_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
