"""DINOv2-эмбеддинги снимков для ансамблевого классификатора срастаний.

Формат совместим с обучением ансамбля: вход 518×518 (прямой даунскейл),
вектор = CLS-токен ⊕ среднее patch-токенов (384+384=768 для ViT-S/14).
Модель кэшируется на процесс; при недоступности timm/весов возвращает None.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_MODEL = None
_TRANSFORM = None
_DEVICE = None
_FAILED = False

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"


def _lazy_init() -> bool:
    global _MODEL, _TRANSFORM, _DEVICE, _FAILED
    if _MODEL is not None:
        return True
    if _FAILED:
        return False
    try:
        import timm
        import torch

        _DEVICE = ("mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        _MODEL.eval().to(_DEVICE)
        cfg = timm.data.resolve_model_data_config(_MODEL)
        _TRANSFORM = (np.array(cfg["mean"], np.float32),
                      np.array(cfg["std"], np.float32))
        return True
    except Exception:
        _FAILED = True
        return False


def dinov2_embed(rgb: np.ndarray) -> Optional[np.ndarray]:
    """768-мерный эмбеддинг кадра (или None, если бэкбон недоступен)."""
    if not _lazy_init():
        return None
    import cv2
    import torch

    mean, std = _TRANSFORM
    img = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - mean) / std
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(_DEVICE)
    with torch.no_grad():
        tokens = _MODEL.forward_features(t)  # (1, 1+n, 384)
        cls = tokens[:, 0]
        patch = tokens[:, 1:].mean(1)
        emb = torch.cat([cls, patch], dim=1)[0].cpu().numpy()
    return emb.astype(np.float32)
