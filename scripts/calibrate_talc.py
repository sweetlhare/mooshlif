"""Калибровка доли талька: изотоническая регрессия raw_frac → true_frac.

Прогоняет обученную модель по pos/neg-примерам, фитит изотонику на train-группах,
оценивает на val-группах (тех же, что при обучении), пишет
models/talc_calibration.json, который подхватывает TalcDetector.

Запуск:
    python scripts/calibrate_talc.py --model models/talc_unet_v3.pt \
        --prep-dir reports/talc_train_data_v3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shlifscan.config import TalcConfig
from shlifscan.talc import TalcDetector

# должен совпадать с train_talc.train (val_frac=0.22, seed 42)
def split_groups(index: list[dict], val_frac: float = 0.22):
    pos_groups = sorted({i["group"] for i in index if i["kind"] == "pos"})
    rng = random.Random(42)
    return set(rng.sample(pos_groups, max(2, int(len(pos_groups) * val_frac))))


def predict_frac(det: TalcDetector, z: dict, thr: float) -> tuple[float, float, float]:
    """(raw_frac, true_frac, iou) на валидной области примера."""
    valid = z["valid"] > 0
    true = (z["eval_target"] > 0) & valid

    class _Pre:  # минимальный Preprocessed-совместимый объект
        pass

    pre = _Pre()
    pre.rgb = z["image"]
    pre.valid = valid
    prob = det._predict_unet(z["image"])
    pred = (prob >= thr) & valid
    pred = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8)).astype(bool)
    raw = float(pred[valid].mean())
    true_frac = float(true[valid].mean())
    inter, union = (pred & true).sum(), (pred | true).sum()
    return raw, true_frac, float(inter / union) if union else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/talc_unet_v3.pt")
    ap.add_argument("--prep-dir", default="reports/talc_train_data_v3")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--out", default="models/talc_calibration.json")
    args = ap.parse_args()

    # калибровочный json не должен применяться при сборе сырых долей
    out_path = Path(args.out)
    backup = None
    if out_path.exists():
        backup = out_path.read_text(encoding="utf-8")
        out_path.unlink()

    cfg = TalcConfig(model_path=args.model, prob_thr=args.thr)
    det = TalcDetector(cfg)
    assert det._model is not None, "модель не загружена"

    prep = Path(args.prep_dir)
    index = json.loads((prep / "index.json").read_text(encoding="utf-8"))
    val_groups = split_groups(index)

    rows = []
    for item in index:
        z = dict(np.load(prep / item["name"]))
        raw, true, iou = predict_frac(det, z, args.thr)
        is_val = item["kind"] == "pos" and item["group"] in val_groups
        rows.append({"name": item["name"], "kind": item["kind"],
                     "val": is_val, "raw": raw, "true": true, "iou": iou})
        print(f"{item['name']:44s} {'VAL' if is_val else '   '} "
              f"raw={raw:.3f} true={true:.3f} iou={iou:.3f}", flush=True)

    train_rows = [r for r in rows if not r["val"]]
    val_rows = [r for r in rows if r["val"]]

    from sklearn.isotonic import IsotonicRegression

    # якоря (0,0) и (1,1): без верхнего якоря изотоника выходит на плато
    # и все высокие доли схлопываются в одну константу
    x = np.array([r["raw"] for r in train_rows] + [0.0, 1.0])
    y = np.array([r["true"] for r in train_rows] + [0.0, 1.0])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)

    def report(rs, tag):
        raw_mae = float(np.mean([abs(r["raw"] - r["true"]) for r in rs]))
        cal_mae = float(np.mean([abs(float(iso.predict([r["raw"]])[0]) - r["true"])
                                 for r in rs]))
        iou = float(np.mean([r["iou"] for r in rs if r["kind"] == "pos"] or [0]))
        print(f"{tag}: raw MAE {raw_mae*100:.2f}% | calib MAE {cal_mae*100:.2f}% "
              f"| pos IoU {iou:.3f}")
        return cal_mae

    report(train_rows, "TRAIN")
    val_mae = report(val_rows, "VAL  ") if val_rows else 0.03

    grid = np.linspace(0, 1, 41)
    calib = {
        "prob_thr": args.thr,
        "isotonic_x": [round(float(v), 4) for v in grid],
        "isotonic_y": [round(float(iso.predict([v])[0]), 4) for v in grid],
        "val_mae": round(val_mae, 4),
        "model": Path(args.model).name,
        "notes": "raw_frac→true_frac; мультискейл 1.0/0.7/0.5, работа на 1536, "
                 "Reinhard к talc_ref_stats.json",
    }
    out_path.write_text(json.dumps(calib, indent=1), encoding="utf-8")
    print("сохранено:", out_path)
    if backup:
        Path(str(out_path) + ".v1_backup").write_text(backup, encoding="utf-8")


if __name__ == "__main__":
    main()
