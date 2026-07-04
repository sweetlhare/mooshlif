"""End-to-end пайплайн: снимок/панорама → фазовая карта → метрики → вердикт."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .classify import (OreMetrics, OreVerdict, decide, load_intergrowth_model,
                       load_talc_vote)
from .config import PipelineConfig
from .imio import imread_rgb, image_size, iter_tiles
from .intergrowth import FEATURE_NAMES, classify_intergrowths, image_level_features
from .preprocess import NormConstants, compute_norm_constants, preprocess
from .segment import adaptive_b_split, segment_phases
from .talc import TalcDetector
from .visualize import PH_FINE, PH_GRAY, PH_ORDINARY, PH_TALC


@dataclass
class AnalysisResult:
    phase_map: np.ndarray        # uint8, метки PH_* (в рабочем масштабе)
    rgb_preview: np.ndarray      # рабочее RGB
    confidence: np.ndarray       # float32 [0..1]
    metrics: OreMetrics
    verdict: OreVerdict
    elapsed_s: float
    scale: float                 # рабочий масштаб относительно исходника
    extra: dict = field(default_factory=dict)  # признаки, пороги — для отчёта


def _analyze_frame(
    rgb: np.ndarray, cfg: PipelineConfig, talc_detector: TalcDetector,
    consts: Optional[NormConstants] = None, work_width: Optional[int] = None,
):
    """Один кадр → (phase_map, confidence, valid, rgb_work, extra)."""
    pre = preprocess(rgb, cfg.preprocess, consts=consts)
    seg = segment_phases(pre, cfg.segment)
    inter = classify_intergrowths(
        seg.sulfide, cfg.intergrowth, work_width=work_width or pre.rgb.shape[1]
    )
    talc_mask, talc_prob = talc_detector.predict_mask(pre)

    phase_map = np.zeros(pre.ln.shape, np.uint8)
    phase_map[seg.gray] = PH_GRAY
    talc_mask &= ~seg.sulfide          # тальк не может быть на сульфидах
    phase_map[talc_mask] = PH_TALC
    phase_map[inter.ordinary] = PH_ORDINARY
    phase_map[inter.fine] = PH_FINE

    conf = seg.confidence.copy()
    talc_conf = np.clip(np.abs(talc_prob - talc_detector.cfg.prob_thr) / 0.5, 0, 1)
    conf = np.minimum(conf, np.where(talc_mask, talc_conf, 1.0).astype(np.float32))

    # признаки срастаний считаются в масштабе КАДРА (для панорамы — тайла):
    # это совпадает с распределением обучающей выборки классификатора
    feats = image_level_features(
        seg.sulfide, cfg.intergrowth.ref_width / pre.rgb.shape[1]
    )
    feats["gray_frac"] = float((seg.gray & pre.valid).sum()) / max(int(pre.valid.sum()), 1)
    extra = {
        "t_split": seg.t_split,
        "image_features": feats,
        "n_aggregates": len(inter.features),
    }
    return phase_map, conf, pre.valid, pre.rgb, extra


def compute_metrics(phase_map: np.ndarray, valid: np.ndarray,
                    confidence: np.ndarray, scale: float) -> OreMetrics:
    area = int(valid.sum())
    if area == 0:
        return OreMetrics()
    pct = lambda m: 100.0 * float(m.sum()) / area  # noqa: E731
    ordinary = pct((phase_map == PH_ORDINARY) & valid)
    fine = pct((phase_map == PH_FINE) & valid)
    talc = pct((phase_map == PH_TALC) & valid)
    gray = pct((phase_map == PH_GRAY) & valid)
    sulf = ordinary + fine
    return OreMetrics(
        sulfide_total_pct=sulf,
        ordinary_pct=ordinary,
        fine_pct=fine,
        talc_pct=talc,
        gray_phase_pct=gray,
        analyzed_area_px=int(area / (scale * scale) if scale else area),
        ordinary_share=(100.0 * ordinary / sulf) if sulf > 0 else 0.0,
        fine_share=(100.0 * fine / sulf) if sulf > 0 else 0.0,
        confidence=float(confidence[valid].mean()),
    )


def _pano_norm_constants(path: str | Path, cfg: PipelineConfig) -> NormConstants:
    """Глобальные константы нормализации панорамы по её даунскейлу.

    Считаются один раз и применяются ко всем тайлам — иначе доли фаз
    «плавают» между тайлами (валидировано EDA: расхождение долей сульфида
    до 1.25× при per-tile нормализации).
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((cfg.pano_norm_side, cfg.pano_norm_side))
        small = np.asarray(im)
    consts = compute_norm_constants(small, cfg.preprocess)
    # t_split тоже фиксируем глобально
    pre = preprocess(small, cfg.preprocess, consts=consts, downscale=False)
    consts.t_split = adaptive_b_split(pre.ln, pre.db, pre.valid)
    return consts


def analyze_image(path: str | Path, cfg: Optional[PipelineConfig] = None,
                  progress: Optional[Callable[[float, str], None]] = None) -> AnalysisResult:
    """Анализ одного снимка или панорамы (автовыбор тайлового режима)."""
    cfg = cfg or PipelineConfig()
    t0 = time.time()
    talc_detector = TalcDetector(cfg.talc)

    w, h = image_size(path)
    is_pano = max(w, h) > cfg.pano_min_side
    extra: dict = {}

    tile_feats: list[tuple[float, dict]] = []  # (вес=площадь сульфидов, признаки)
    tile_embs: list[tuple[float, "np.ndarray"]] = []
    vote_embs: list["np.ndarray"] = []         # ВСЕ тайлы — для голоса талька

    from .embeddings import dinov2_embed

    if not is_pano:
        rgb = imread_rgb(path)
        phase_map, conf, valid, rgb_work, extra = _analyze_frame(rgb, cfg, talc_detector)
        scale = rgb_work.shape[1] / w
        tile_feats.append((max(extra["image_features"]["sulf_frac"], 1e-6), extra["image_features"]))
        emb = dinov2_embed(rgb_work)
        if emb is not None:
            tile_embs.append((1.0, emb))
            vote_embs.append(emb)
    else:
        consts = _pano_norm_constants(path, cfg)
        extra["t_split"] = consts.t_split
        # выходной масштаб: тайл pano_tile -> analysis_max_side
        scale = cfg.preprocess.analysis_max_side / cfg.pano_tile
        out_w, out_h = int(round(w * scale)), int(round(h * scale))
        phase_map = np.zeros((out_h, out_w), np.uint8)
        conf = np.zeros((out_h, out_w), np.float32)
        valid = np.zeros((out_h, out_w), bool)
        rgb_work = np.zeros((out_h, out_w, 3), np.uint8)

        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            im = im.convert("RGB")
            tiles = list(iter_tiles(w, h, cfg.pano_tile, cfg.pano_overlap))
            # эффективная ширина всей панорамы в рабочем масштабе тайла —
            # для корректной нормировки толщин к ref_width
            work_width = int(w * scale)
            for i, t in enumerate(tiles):
                tile_rgb = np.asarray(im.crop((t.x, t.y, t.x + t.w, t.y + t.h)))
                pm, cf, vd, rw, tile_extra = _analyze_frame(
                    tile_rgb, cfg, talc_detector, consts=consts, work_width=work_width
                )
                tf = tile_extra["image_features"]
                emb = dinov2_embed(rw)
                if emb is not None:
                    vote_embs.append(emb)
                if tf["sulf_frac"] > 0.001:
                    tile_feats.append((tf["sulf_frac"], tf))
                    if emb is not None:
                        tile_embs.append((tf["sulf_frac"], emb))
                sx = rw.shape[1] / t.w
                ox0 = int(round(t.core_x * scale)); oy0 = int(round(t.core_y * scale))
                ox1 = min(int(round((t.core_x + t.core_w) * scale)), out_w)
                oy1 = min(int(round((t.core_y + t.core_h) * scale)), out_h)
                cx0 = int(round((t.core_x - t.x) * sx)); cy0 = int(round((t.core_y - t.y) * sx))
                # конец ядра в координатах проанализированного тайла — берём РОВНО
                # ядро (без правого/нижнего перекрытия), иначе контент сжимается
                # и маска уезжает вверх-влево относительно исходника
                cx1 = int(round((t.core_x - t.x + t.core_w) * sx))
                cy1 = int(round((t.core_y - t.y + t.core_h) * sx))
                cw, ch = ox1 - ox0, oy1 - oy0
                if cw <= 0 or ch <= 0:
                    continue

                def fit(arr, interp):
                    return cv2.resize(arr[cy0:cy1, cx0:cx1], (cw, ch), interpolation=interp)

                phase_map[oy0:oy1, ox0:ox1] = fit(pm, cv2.INTER_NEAREST)
                conf[oy0:oy1, ox0:ox1] = fit(cf, cv2.INTER_LINEAR)
                valid[oy0:oy1, ox0:ox1] = fit(vd.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
                rgb_work[oy0:oy1, ox0:ox1] = fit(rw, cv2.INTER_AREA)
                if progress:
                    progress((i + 1) / len(tiles), f"тайл {i + 1}/{len(tiles)}")

    metrics = compute_metrics(phase_map, valid, conf, scale)
    # калибровка доли талька (изотоника поверх сырой площади маски)
    raw_talc_pct = metrics.talc_pct
    metrics.talc_pct = round(
        100.0 * talc_detector.calibrate_fraction(raw_talc_pct / 100.0), 2
    )
    extra["talc_pct_raw"] = round(raw_talc_pct, 2)
    extra["talc_ci_half_pct"] = talc_detector.frac_ci_half_pct

    # агрегированные признаки срастаний (взвешенно по площади сульфидов в кадрах)
    agg_feats: dict = {}
    if tile_feats:
        wsum = sum(wt for wt, _ in tile_feats)
        all_keys = FEATURE_NAMES + ["gray_frac"]
        agg_feats = {
            k: float(sum(wt * f.get(k, 0.0) for wt, f in tile_feats) / wsum)
            for k in all_keys
        }
    extra["image_features"] = agg_feats

    agg_emb = None
    if tile_embs:
        wsum = sum(wt for wt, _ in tile_embs)
        agg_emb = sum(wt * e for wt, e in tile_embs) / wsum

    # голос оталькованности: средняя per-tile вероятность по ВСЕМ тайлам
    # (голос обучен на одиночных снимках — усреднение вероятностей, а не
    # эмбеддингов, сохраняет распределение входов модели)
    talc_vote = load_talc_vote()
    vote_prob = None
    if talc_vote is not None and vote_embs:
        try:
            probs = talc_vote["model"].predict_proba(np.stack(vote_embs))[:, 1]
            vote_prob = float(np.mean(probs))
            extra["talc_vote_tiles"] = [round(float(p), 3) for p in probs[:64]]
        except Exception:
            vote_prob = None

    model_bundle = load_intergrowth_model() if agg_feats else None
    verdict = decide(
        metrics, cfg.talc.talc_ore_thr_pct,
        features=agg_feats or None, model_bundle=model_bundle,
        embedding=agg_emb, talc_vote=talc_vote, talc_vote_prob=vote_prob,
        talc_frac_vote_gate=cfg.talc.talc_frac_vote_gate,
    )
    return AnalysisResult(
        phase_map=phase_map, rgb_preview=rgb_work, confidence=conf,
        metrics=metrics, verdict=verdict,
        elapsed_s=time.time() - t0, scale=scale, extra=extra,
    )
