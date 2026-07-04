"""Качественная проверка на эталонных рисунках организаторов:
находит ли модель тальк там, куда указывают стрелки «Тальк» в
«Постановке задачи» (docs/org_reference/image4.jpeg, image5.jpeg).

Стрелки/подписи (чисто синие пиксели) маскируются перед инференсом.
Выход: триптихи оригинал | вероятность талька | оверлей маски.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shlifscan.config import PipelineConfig, TalcConfig
from shlifscan.imio import imread_rgb
from shlifscan.preprocess import detect_annotation_lines, preprocess
from shlifscan.segment import segment_phases
from shlifscan.talc import TalcDetector

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "docs" / "org_reference"
OUT = ROOT / "reports" / "org_reference_check"


def inpaint_annotations(rgb: np.ndarray) -> np.ndarray:
    """Закрашивает синие стрелки/подписи перед анализом."""
    lines = detect_annotation_lines(rgb)
    if not lines.any():
        return rgb
    return cv2.inpaint(rgb, lines.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)


def main(model_path: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    det = TalcDetector(TalcConfig(model_path=model_path))
    print("модель:", model_path, "| U-Net загружена:", det._model is not None)

    for name in ["image4.jpeg", "image5.jpeg", "image2.png", "image3.jpeg"]:
        f = REF / name
        if not f.exists():
            continue
        rgb = inpaint_annotations(imread_rgb(f))
        pre = preprocess(rgb, cfg.preprocess, downscale=False)
        seg = segment_phases(pre, cfg.segment)
        prob = det.predict_proba(pre)
        mask = (prob >= det.cfg.prob_thr) & pre.valid & ~seg.sulfide & ~seg.gray

        heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        ovl = rgb.copy()
        blue = np.zeros_like(rgb); blue[..., 2] = 255
        a = 0.45
        ovl[mask] = (a * blue[mask] + (1 - a) * ovl[mask]).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ovl, contours, -1, (79, 142, 247), 2)

        trip = np.hstack([rgb, heat, ovl])
        cv2.imwrite(str(OUT / f"check_{f.stem}.jpg"),
                    cv2.cvtColor(trip, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        frac = float(mask[pre.valid].mean()) * 100
        cal = det.calibrate_fraction(frac / 100) * 100
        print(f"{name}: тальк {frac:.1f}% (сырой) / {cal:.1f}% (калибр.)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/talc_unet_v5.pt")
