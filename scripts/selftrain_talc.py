"""Self-training талька: псевдоразметка неразмеченного тёмного домена
моделью v3 → дообучение с rehearsal (рецепт: классово-адаптивные пороги
уверенности, ignore для неуверенного, старые примеры сохраняются).

Запуск:
    python scripts/selftrain_talc.py --model models/talc_unet_v3.pt \
        --base-prep reports/talc_train_data_v3 --out-prep reports/talc_train_data_v4st
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shlifscan.config import PipelineConfig, TalcConfig
from shlifscan.imio import imread_rgb
from shlifscan.preprocess import preprocess, reinhard_to_reference
from shlifscan.segment import segment_phases
from shlifscan.talc import TalcDetector

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK_SIDE = 1536

# классово-адаптивные пороги (тальк — редкий класс, порог мягче)
POS_THR = 0.75
NEG_THR = 0.12


def pseudo_example(rgb: np.ndarray, det: TalcDetector, ref: dict,
                   cfg: PipelineConfig):
    """Строит псевдоразмеченный пример или None, если мало уверенных пикселей."""
    h, w = rgb.shape[:2]
    s = WORK_SIDE / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    pre = preprocess(rgb, cfg.preprocess, downscale=False)
    seg = segment_phases(pre, cfg.segment)
    bright = seg.sulfide | seg.gray

    prob = det._predict_unet(rgb)
    target = np.zeros(prob.shape, np.uint8)
    weight = np.zeros(prob.shape, np.float32)

    pos = (prob >= POS_THR) & ~bright & pre.valid
    neg = ((prob <= NEG_THR) | bright) & pre.valid
    target[pos] = 1
    weight[pos] = 0.7          # псевдопозитив — с осторожным весом
    weight[neg & ~pos] = 0.5   # псевдонегатив
    weight[bright & pre.valid] = 1.0  # яркие фазы — надёжный негатив

    pos_frac = float(pos.mean())
    if pos_frac < 0.005:  # почти нет уверенного талька — пример малополезен
        weight[:] = np.where(bright & pre.valid, 1.0, weight * 0.5)

    rgb_norm = reinhard_to_reference(rgb, ref)
    return {
        "image": rgb_norm,
        "target": target,
        "weight": weight,
        "eval_target": target,  # псевдо: не для валидации
        "valid": pre.valid.astype(np.uint8),
        "pos_frac": pos_frac,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/talc_unet_v3.pt")
    ap.add_argument("--base-prep", default="reports/talc_train_data_v3")
    ap.add_argument("--out-prep", default="reports/talc_train_data_v4st")
    ap.add_argument("--n-ch2", type=int, default=40)
    ap.add_argument("--n-pano-tiles", type=int, default=36)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = Path(args.base_prep)
    out = Path(args.out_prep)
    out.mkdir(parents=True, exist_ok=True)

    # rehearsal: копируем все базовые примеры
    index = json.loads((base / "index.json").read_text(encoding="utf-8"))
    for item in index:
        shutil.copy2(base / item["name"], out / item["name"])
    print(f"rehearsal: {len(index)} базовых примеров")

    ref = json.loads((ROOT / "models" / "talc_ref_stats.json").read_text())
    cfg = PipelineConfig()
    det = TalcDetector(TalcConfig(model_path=args.model))
    assert det._model is not None

    # --- ч2 оталькованные (тёмный домен, есть тальк по метке папки) ---
    import hashlib

    ch2_dir = DATA / "Фото руд по сортам. ч2/оталькованные"
    annot_dir = DATA / "Фото руд по сортам. ч1/Оталькованные руды"
    pos_md5 = {hashlib.md5(f.read_bytes()).hexdigest()
               for f in annot_dir.glob("*.JPG")}
    files = [f for f in sorted(ch2_dir.iterdir())
             if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
             and hashlib.md5(f.read_bytes()).hexdigest() not in pos_md5]
    files = rng.sample(files, min(args.n_ch2, len(files)))
    n_added = 0
    for f in files:
        ex = pseudo_example(imread_rgb(f), det, ref, cfg)
        name = f"pseudo_ch2_{f.stem}.npz".replace(" ", "_")
        np.savez_compressed(out / name, **{k: v for k, v in ex.items() if k != "pos_frac"})
        index.append({"name": name, "kind": "neg",  # kind neg => не попадёт в val-сплит
                      "group": "pseudo_" + f.stem, "pseudo": True,
                      "pos_frac": round(ex["pos_frac"], 4)})
        n_added += 1
        print(f"ch2 {f.name}: уверенный тальк {ex['pos_frac']*100:.1f}%", flush=True)

    # --- тайлы панорам ---
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    panos = sorted((DATA / "Панорамы").glob("*.jpg"))
    per_pano = max(args.n_pano_tiles // len(panos), 2)
    for pf in panos:
        with Image.open(pf) as im:
            im = im.convert("RGB")
            W, H = im.size
            for i in range(per_pano):
                x = rng.randint(0, max(W - 3072, 1))
                y = rng.randint(0, max(H - 3072, 1))
                tile = np.asarray(im.crop((x, y, x + 3072, y + 3072)))
                if tile.std() < 8:   # пустое поле/скол
                    continue
                ex = pseudo_example(tile, det, ref, cfg)
                name = f"pseudo_pano{pf.stem}_{i}.npz"
                np.savez_compressed(out / name,
                                    **{k: v for k, v in ex.items() if k != "pos_frac"})
                index.append({"name": name, "kind": "neg",
                              "group": f"pseudo_pano{pf.stem}", "pseudo": True,
                              "pos_frac": round(ex["pos_frac"], 4)})
                n_added += 1
        print(f"панорама {pf.name}: тайлы добавлены", flush=True)

    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"итого: +{n_added} псевдопримеров → {out}")


if __name__ == "__main__":
    main()
