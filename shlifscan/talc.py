"""Детекция талька: U-Net (обученная на слабой разметке) или классический детектор.

Инференс-рецепт (валидирован диагностикой):
- Reinhard-нормализация LAB к статистикам обучающей выборки (перенос доменов);
- рабочая длинная сторона 1536 (как при обучении);
- усреднение вероятностей по масштабам 1.0/0.7/0.5 (масштабная устойчивость);
- изотоническая калибровка ДОЛИ талька (models/talc_calibration.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import TalcConfig
from .preprocess import Preprocessed, reinhard_to_reference

WORK_SIDE = 1536
MULTISCALE = (1.0, 0.7, 0.5)


class TalcDetector:
    """Возвращает попиксельную вероятность талька [0..1] и калиброванную долю."""

    def __init__(self, cfg: TalcConfig):
        self.cfg = cfg
        self._model = None
        self._device = None
        self._ref_stats: Optional[dict] = None
        self._calib: Optional[dict] = None
        if cfg.model_path and Path(cfg.model_path).exists():
            try:
                self._load_model(cfg.model_path)
            except Exception as e:  # noqa: BLE001 — не роняем анализ из-за модели
                import sys
                print(f"[talc] не удалось загрузить {cfg.model_path}: {e!r}; "
                      "переключаюсь на классическую детекцию", file=sys.stderr)
                self._model = None

    # ------------------------------------------------------------------ U-Net
    def _load_model(self, path: str) -> None:
        import torch

        self._device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        if path.endswith(".ts"):
            self._model = torch.jit.load(path, map_location=self._device)
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                import segmentation_models_pytorch as smp

                self._model = smp.Unet(
                    ckpt.get("encoder", "resnet34"),
                    encoder_weights=None,
                    classes=ckpt.get("classes", 1),
                )
                self._model.load_state_dict(ckpt["state_dict"])
                self._model.to(self._device)
            else:
                self._model = ckpt.to(self._device)
        self._model.eval()

        mdir = Path(path).parent
        ref = mdir / "talc_ref_stats.json"
        if ref.exists():
            self._ref_stats = json.loads(ref.read_text(encoding="utf-8"))
        calib = mdir / "talc_calibration.json"
        if calib.exists():
            self._calib = json.loads(calib.read_text(encoding="utf-8"))
            if "prob_thr" in self._calib:
                self.cfg.prob_thr = float(self._calib["prob_thr"])

    def _forward_tiles(self, img: np.ndarray) -> np.ndarray:
        """U-Net по тайлам; img — float32 [0..1] HxWx3."""
        import torch

        t, ov = self.cfg.tile, self.cfg.overlap
        h, w = img.shape[:2]
        prob = np.zeros((h, w), np.float32)
        weight = np.zeros((h, w), np.float32)
        step = t - ov
        with torch.no_grad():
            for y in range(0, max(h - ov, 1), step):
                for x in range(0, max(w - ov, 1), step):
                    y1, x1 = min(y + t, h), min(x + t, w)
                    y0, x0 = max(y1 - t, 0), max(x1 - t, 0)
                    patch = img[y0:y1, x0:x1]
                    ph, pw = patch.shape[:2]
                    if ph < t or pw < t:
                        patch = np.pad(
                            patch, ((0, t - ph), (0, t - pw), (0, 0)), mode="reflect"
                        )
                    inp = torch.from_numpy(patch.transpose(2, 0, 1))[None].to(self._device)
                    out = torch.sigmoid(self._model(inp))[0, 0].cpu().numpy()[:ph, :pw]
                    prob[y0:y0 + ph, x0:x0 + pw] += out
                    weight[y0:y0 + ph, x0:x0 + pw] += 1.0
        return prob / np.maximum(weight, 1e-6)

    def _predict_unet(self, rgb: np.ndarray) -> np.ndarray:
        h0, w0 = rgb.shape[:2]
        if self._ref_stats is not None:
            rgb = reinhard_to_reference(rgb, self._ref_stats)
        # рабочее разрешение обучения
        s = WORK_SIDE / max(h0, w0)
        base = cv2.resize(rgb, (int(w0 * s), int(h0 * s)),
                          interpolation=cv2.INTER_AREA) if s < 1 else rgb
        bh, bw = base.shape[:2]
        acc = np.zeros((bh, bw), np.float32)
        for ms in MULTISCALE:
            im = base if ms == 1.0 else cv2.resize(
                base, (max(int(bw * ms), 64), max(int(bh * ms), 64)),
                interpolation=cv2.INTER_AREA)
            p = self._forward_tiles(im.astype(np.float32) / 255.0)
            if ms != 1.0:
                p = cv2.resize(p, (bw, bh), interpolation=cv2.INTER_LINEAR)
            acc += p
        acc /= len(MULTISCALE)
        if (bh, bw) != (h0, w0):
            acc = cv2.resize(acc, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return acc

    # -------------------------------------------------------------- классика
    @staticmethod
    def _predict_classic(pre: Preprocessed) -> np.ndarray:
        """Фолбэк-эвристика: тёмные области матрицы с мелкозернистой текстурой."""
        l = pre.ln
        mean = cv2.boxFilter(l, -1, (15, 15))
        sq_mean = cv2.boxFilter(l * l, -1, (15, 15))
        local_std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
        darkness = np.clip((0.45 - l) / 0.45, 0, 1)
        texture = np.clip(local_std / 0.08, 0, 1)
        prob = cv2.GaussianBlur(darkness * texture, (0, 0), 5)
        prob[~pre.valid] = 0.0
        return prob.astype(np.float32)

    # ------------------------------------------------------------------- API
    def predict_proba(self, pre: Preprocessed) -> np.ndarray:
        if self._model is not None:
            return self._predict_unet(pre.rgb)
        return self._predict_classic(pre)

    def predict_mask(self, pre: Preprocessed) -> tuple[np.ndarray, np.ndarray]:
        prob = self.predict_proba(pre)
        mask = (prob >= self.cfg.prob_thr) & pre.valid
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        ).astype(bool)
        return mask, prob

    def calibrate_fraction(self, raw_frac: float) -> float:
        """Изотоническая калибровка доли талька (raw → калиброванная)."""
        if not self._calib or "isotonic_x" not in self._calib:
            return raw_frac
        return float(np.interp(
            raw_frac, self._calib["isotonic_x"], self._calib["isotonic_y"]
        ))

    @property
    def frac_ci_half_pct(self) -> float:
        """Полуширина ДИ доли талька, п.п. (из манифеста калибровки)."""
        if self._calib and "val_mae" in self._calib:
            return round(float(self._calib["val_mae"]) * 100.0, 1)
        return 3.0
