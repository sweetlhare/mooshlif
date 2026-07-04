"""Квантификация доли талька (семейство QuaPy: CC / ACC / PACC / SLD-EMQ)
против изотонической калибровки.

Каждый снимок — «мешок» пикселей; U-Net даёт soft-вероятность талька на пиксель.
Долю талька (prevalence) оцениваем методами квантификации, которые
bias-корректируют систематическую ПЕРЕоценку классификатора:

  CC   — Classify & Count: доля пикселей prob≥thr (наш raw)
  PCC  — Probabilistic CC: средняя вероятность
  ACC  — Adjusted CC:  (CC − FPR)/(TPR − FPR),  TPR/FPR по порогу на train
  PACC — Probabilistic ACC: то же на СОФТ tpr/fpr (средние вероятности по классам)
  SLD  — Saerens-Latinne-Decaestecker EM (адаптация априори по-снимочно)

TPR/FPR оцениваются на train-пикселях (метка = eval_target). Оценка val MAE —
против доли eval_target, тот же таргет, что у изотоники → сравнение честное.
Запуск: python scripts/quantify_talc.py --model models/talc_unet.pt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shlifscan.config import TalcConfig
from shlifscan.talc import TalcDetector

ROOT = Path(__file__).resolve().parents[1]


def split_groups(index, val_frac=0.22):
    pos = sorted({i["group"] for i in index if i["kind"] == "pos"})
    rng = random.Random(42)
    return set(rng.sample(pos, max(2, int(len(pos) * val_frac))))


def sld_emq(probs, prior0, n_iter=50):
    """SLD/EMQ: EM-адаптация априори талька в мешке пикселей.
    probs — P(talc|pixel) при train-априори prior0. Возвращает адаптированную долю."""
    p = np.clip(prior0, 1e-4, 1 - 1e-4)
    pr = np.clip(probs, 1e-6, 1 - 1e-6)
    for _ in range(n_iter):
        # пересчёт постериоров при текущем априори p (отношение к train-априори prior0)
        num = pr * (p / prior0)
        den = num + (1 - pr) * ((1 - p) / (1 - prior0))
        post = num / np.clip(den, 1e-9, None)
        new_p = float(post.mean())
        if abs(new_p - p) < 1e-5:
            p = new_p
            break
        p = np.clip(new_p, 1e-4, 1 - 1e-4)
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/talc_unet.pt")
    ap.add_argument("--prep-dir", default="reports/talc_train_data_v5")
    ap.add_argument("--thr", type=float, default=0.4)
    args = ap.parse_args()

    det = TalcDetector(TalcConfig(model_path=args.model, prob_thr=args.thr))
    assert det._model is not None
    prep = Path(args.prep_dir)
    index = json.loads((prep / "index.json").read_text(encoding="utf-8"))
    val_groups = split_groups(index)

    rows = []
    print("инференс по примерам…", flush=True)
    for it in index:
        z = dict(np.load(prep / it["name"]))
        valid = z["valid"] > 0
        true = (z["eval_target"] > 0) & valid
        prob = det._predict_unet(z["image"])
        pv = prob[valid]
        rows.append({
            "name": it["name"],
            "val": it["kind"] == "pos" and it["group"] in val_groups,
            "kind": it["kind"],
            "probs": pv.astype(np.float32),
            "true_pix": true[valid].astype(np.bool_),
            "true_frac": float(true[valid].mean()) if valid.any() else 0.0,
            "cc": float((pv >= args.thr).mean()),
            "pcc": float(pv.mean()),
        })

    train = [r for r in rows if not r["val"]]
    val = [r for r in rows if r["val"]]

    # TPR/FPR по train-пикселям
    tp = np.concatenate([r["probs"] for r in train])
    tl = np.concatenate([r["true_pix"] for r in train])
    thr = args.thr
    tpr = float((tp[tl] >= thr).mean()); fpr = float((tp[~tl] >= thr).mean())
    stpr = float(tp[tl].mean()); sfpr = float(tp[~tl].mean())
    prior0 = float(tl.mean())
    print(f"train: TPR={tpr:.3f} FPR={fpr:.3f} | soft tpr={stpr:.3f} fpr={sfpr:.3f} "
          f"| априори талька={prior0:.3f}")

    def acc(cc):
        d = tpr - fpr
        return float(np.clip((cc - fpr) / d, 0, 1)) if abs(d) > 1e-6 else cc

    def pacc(pcc):
        d = stpr - sfpr
        return float(np.clip((pcc - sfpr) / d, 0, 1)) if abs(d) > 1e-6 else pcc

    # изотоника из прод-калибровки (для сравнения)
    calib = json.loads((ROOT / "models/talc_calibration.json").read_text(encoding="utf-8"))
    xs, ys = np.array(calib["isotonic_x"]), np.array(calib["isotonic_y"])
    iso = lambda v: float(np.interp(v, xs, ys))

    def report(rs, tag):
        def mae(fn):
            return float(np.mean([abs(fn(r) - r["true_frac"]) for r in rs])) * 100
        m_cc = mae(lambda r: r["cc"])
        m_iso = mae(lambda r: iso(r["cc"]))
        m_acc = mae(lambda r: acc(r["cc"]))
        m_pacc = mae(lambda r: pacc(r["pcc"]))
        m_sld = mae(lambda r: sld_emq(r["probs"], prior0))
        print(f"\n{tag} (n={len(rs)}):")
        print(f"  CC (raw)      MAE {m_cc:5.2f}%")
        print(f"  ИЗОТОНИКА     MAE {m_iso:5.2f}%   ← прод")
        print(f"  ACC           MAE {m_acc:5.2f}%")
        print(f"  PACC          MAE {m_pacc:5.2f}%")
        print(f"  SLD/EMQ       MAE {m_sld:5.2f}%")

    report(train, "TRAIN")
    if val:
        report(val, "VAL")


if __name__ == "__main__":
    main()
