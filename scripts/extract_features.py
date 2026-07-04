"""Быстрое извлечение признаков срастаний по всем размеченным снимкам
(без талька и попиксельных агрегатов) — для обучения классификатора руды.

Запуск:
    python scripts/extract_features.py --out reports/features_v2.csv
"""

from __future__ import annotations

import argparse
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shlifscan.config import PipelineConfig
from shlifscan.imio import imread_rgb
from shlifscan.intergrowth import image_level_features
from shlifscan.preprocess import preprocess
from shlifscan.segment import segment_phases

from validate_classification import FOLDERS, list_images  # noqa: E402


def sample_group(name: str) -> str:
    stem = Path(name).stem
    for sep in (" ", "-", "_"):
        if sep in stem:
            stem = stem.split(sep)[0]
    return stem


def process(item: dict) -> dict:
    cfg = PipelineConfig()
    try:
        rgb = imread_rgb(item["path"])
        pre = preprocess(rgb, cfg.preprocess)
        seg = segment_phases(pre, cfg.segment)
        width_scale = cfg.intergrowth.ref_width / pre.rgb.shape[1]
        feats = image_level_features(seg.sulfide, width_scale)
        valid = pre.valid
        area = max(int(valid.sum()), 1)
        return {
            **item,
            "group": sample_group(item["path"]),
            **{k: round(v, 5) for k, v in feats.items()},
            "gray_frac": round(float((seg.gray & valid).sum()) / area, 5),
            "t_split": round(seg.t_split, 2),
            "b_ref": round(pre.consts.b_ref, 2),
            "error": "",
        }
    except Exception as e:
        traceback.print_exc()
        return {**item, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/features_v2.csv")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = list_images()
    print(f"снимков: {len(items)}", flush=True)
    with Pool(args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(process, items, chunksize=4)):
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(items)}", flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
