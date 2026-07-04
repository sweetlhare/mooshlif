"""Выполнение анализа в отдельном процессе + генерация всех артефактов.

Прогресс пишется в runs/{id}/progress.json, итог — в meta.json.
Функция run_analysis запускается в spawn-процессе (MPS-безопасно).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _progress_writer(run_dir: Path):
    state = {"stage": "старт", "percent": 0}

    def cb(frac: float | None = None, stage: str | None = None,
           base: float = 0.0, span: float = 100.0):
        if stage:
            state["stage"] = stage
        if frac is not None:
            state["percent"] = int(base + span * frac)
        _write(run_dir / "progress.json", state)

    return cb


def run_analysis(run_id: str, runs_root: str, src_path: str, params_json: str) -> None:
    """Точка входа воркера (вызывается в отдельном процессе)."""
    import cv2
    import numpy as np

    from app.backend.dzi import make_dzi_from_array, make_dzi_from_file
    from shlifscan.classify import conclusion_text
    from shlifscan.config import PipelineConfig
    from shlifscan.imio import image_size
    from shlifscan.intergrowth import granulometry_curve
    from shlifscan.pipeline import analyze_image
    from shlifscan.report import save_pdf_report, verdict_row
    from shlifscan.visualize import (PH_FINE, PH_ORDINARY, PH_TALC, COLORS,
                                     confidence_heatmap, overlay)

    run_dir = Path(runs_root) / run_id
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    progress = _progress_writer(run_dir)
    t0 = time.time()

    try:
        cfg = PipelineConfig()
        # дефолт продукта: модель талька, если обучена
        default_talc = ROOT / "models" / "talc_unet.pt"
        if default_talc.exists():
            cfg.talc.model_path = str(default_talc)
        overrides = json.loads(params_json) if params_json else {}
        if "talc" in overrides:
            for k, v in overrides["talc"].items():
                setattr(cfg.talc, k, v)

        meta["status"] = "running"
        _write(meta_path, meta)
        progress(0.02, "подготовка")

        w, h = image_size(src_path)
        res = analyze_image(
            src_path, cfg,
            progress=lambda f, msg: progress(f, f"сегментация: {msg}", base=5, span=55),
        )

        progress(0.0, "визуализация", base=60, span=10)
        ovl = overlay(res.rgb_preview, res.phase_map, alpha=0.5)
        heat = confidence_heatmap(res.rgb_preview, res.confidence)
        cv2.imwrite(str(run_dir / "confidence.jpg"),
                    cv2.cvtColor(heat, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        prev = res.rgb_preview
        if prev.shape[1] > 1600:
            s = 1600 / prev.shape[1]
            prev = cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(run_dir / "preview.jpg"),
                    cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(run_dir / "phase_map.png"), res.phase_map)

        # RGBA-слой фаз для DZI
        rgba = np.zeros((*res.phase_map.shape, 4), np.uint8)
        for ph in (PH_ORDINARY, PH_FINE, PH_TALC):
            m = res.phase_map == ph
            rgba[m, :3] = COLORS[ph]
            rgba[m, 3] = 170

        progress(0.0, "пирамиды DZI", base=70, span=20)
        make_dzi_from_file(src_path, run_dir / "dzi" / "image", fmt="jpg")
        progress(0.5, "пирамиды DZI", base=70, span=20)
        make_dzi_from_array(rgba, run_dir / "dzi" / "phases", fmt="png")

        progress(0.0, "отчёты", base=90, span=9)
        sulf_mask = (res.phase_map == PH_ORDINARY) | (res.phase_map == PH_FINE)
        curve = granulometry_curve(sulf_mask)
        m = res.metrics

        # стереология: 95% CI доли талька по межпольной дисперсии (ASTM E562)
        # + модельная компонента (MAE калибровки); MSWD — флаг неоднородности
        from shlifscan.stereology import association_index, fraction_ci
        from shlifscan.visualize import PH_GRAY as _PH_GRAY

        valid_mask = res.confidence > 0
        talc_mask = res.phase_map == PH_TALC
        ci = fraction_ci(
            talc_mask, valid_mask,
            model_mae_pct=float(res.extra.get("talc_ci_half_pct") or 3.0),
            grid=8,
        )
        assoc_talc = association_index(sulf_mask, talc_mask, valid_mask)
        assoc_gray = association_index(sulf_mask, res.phase_map == _PH_GRAY, valid_mask)
        # интервал центрируем на калиброванной доле; полуширина —
        # свёртка пространственной (E562) и модельной компонент
        half = float(np.hypot(ci.spatial_ci_pct, ci.model_ci_pct))
        talc_ci = [round(max(m.talc_pct - half, 0), 2),
                   round(min(m.talc_pct + half, 100), 2)]
        result = {
            "ore_class": res.verdict.ore_class,
            "conclusion": conclusion_text(res.verdict),
            "confidence": round(m.confidence, 3),
            "metrics": {
                "sulfide_total_pct": round(m.sulfide_total_pct, 2),
                "ordinary_pct": round(m.ordinary_pct, 2),
                "fine_pct": round(m.fine_pct, 2),
                "ordinary_share": round(m.ordinary_share, 1),
                "fine_share": round(m.fine_share, 1),
                "talc_pct": round(m.talc_pct, 2),
                "talc_ci": talc_ci,
                "gray_phase_pct": round(m.gray_phase_pct, 2),
            },
            "stereology": {
                "method": "Delesse (Aa=Vv), межпольный CI по ASTM E562, "
                          "MSWD-тест неоднородности",
                "n_fields": ci.n_fields,
                "spatial_ci_pct": ci.spatial_ci_pct,
                "model_ci_pct": ci.model_ci_pct,
                "mswd": ci.mswd,
                "talc_heterogeneous": ci.heterogeneous,
                "association_sulfide_talc": assoc_talc,
                "association_sulfide_gray": assoc_gray,
            },
            "granulometry": curve,
            "features": {k: round(v, 4) for k, v in
                         (res.extra.get("image_features") or {}).items()},
            "explanation": _explanation(res),
            "params": {**{k: v for k, v in res.verdict.params.items()
                          if k != "talc_thr_pct"},
                       "talc_ore_thr_pct": res.verdict.params.get(
                           "talc_thr_pct", cfg.talc.talc_ore_thr_pct),
                       "talc_model": Path(cfg.talc.model_path).name
                       if cfg.talc.model_path else "classic",
                       "config": json.loads(cfg.to_json())},
        }

        save_pdf_report(run_dir / "report.pdf", meta["file_name"], res.verdict,
                        res.rgb_preview, ovl)
        row = verdict_row(meta["file_name"], res.verdict)
        import pandas as pd

        pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False,
                                   encoding="utf-8-sig")

        # широкий assay-профиль под LIMS ГОКов (acQuire/Geobank/Fusion):
        # одна строка = один образец, плоские колонки с единицами в имени
        assay = {
            "SAMPLE_ID": Path(meta["file_name"]).stem,
            "ORE_CLASS": res.verdict.ore_class,
            "SULFIDE_PCT": round(m.sulfide_total_pct, 2),
            "SULF_ORDINARY_PCT": round(m.ordinary_pct, 2),
            "SULF_FINE_PCT": round(m.fine_pct, 2),
            "GRAY_PHASE_PCT": round(m.gray_phase_pct, 2),
            "TALC_PCT": round(m.talc_pct, 2),
            "TALC_CI_LOW_PCT": talc_ci[0],
            "TALC_CI_HIGH_PCT": talc_ci[1],
            "TALC_MSWD": ci.mswd,
            "ECD_P50_PX": curve.get("ecd_p50", 0),
            "ECD_P80_PX": curve.get("ecd_p80", 0),
            "ASSOC_SULF_TALC_IDX": assoc_talc,
            "ASSOC_SULF_GRAY_IDX": assoc_gray,
            "CONFIDENCE": round(m.confidence, 3),
            "ANALYSIS_ID": run_id,
            "ANALYSIS_DATE": meta["created_at"],
        }
        pd.DataFrame([assay]).to_csv(run_dir / "assay.csv", index=False,
                                     encoding="utf-8-sig")
        _export_geojson(res.phase_map, 1.0 / res.scale, run_dir / "mask.geojson")

        meta.update({
            "status": "done",
            "elapsed_s": round(time.time() - t0, 1),
            "image": {"width": w, "height": h, "work_scale": round(res.scale, 4)},
            "result": result,
        })
        _write(meta_path, meta)
        progress(1.0, "done", base=0, span=100)
    except Exception as e:
        traceback.print_exc()
        meta.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
        _write(meta_path, meta)
        progress(1.0, "error")


def _talc_ci_half() -> float:
    """Полуширина ДИ доли талька из манифеста модели (val MAE), иначе консерватив."""
    ckpt = ROOT / "models" / "talc_unet.pt"
    try:
        import torch

        hist = torch.load(ckpt, map_location="cpu", weights_only=False).get("history", [])
        maes = [h["val_frac_mae"] for h in hist]
        return round(min(maes) * 100 * 1.5, 2) if maes else 3.0
    except Exception:
        return 3.0


def _explanation(res) -> list[dict]:
    """Факторы решения для панели «Объяснение»."""
    m = res.metrics
    v = res.verdict
    feats = res.extra.get("image_features") or {}
    out = []
    thr = v.params.get("talc_thr_pct", 10.0)
    vote = v.params.get("talc_vote_prob")
    if v.ore_class == "оталькованная":
        if m.talc_pct > thr:
            factor = f"Доля талька {m.talc_pct:.1f}% > порога {thr:.0f}%"
            if vote is not None:
                factor += f", подтверждено image-level моделью (p={vote:.0%})"
            out.append({"factor": factor, "weight": "решающий"})
        else:
            out.append({"factor": "Визуальный паттерн оталькованной руды "
                                  f"(image-level модель, p={vote or 0:.0%})",
                        "weight": "решающий"})
            out.append({"factor": f"Измеренная доля талька {m.talc_pct:.1f}% ≤ {thr:.0f}% "
                                  "(тёмная съёмка скрывает текстуру талька)",
                        "weight": "справочный"})
    else:
        # доля могла превысить порог, но image-level голос не подтвердил —
        # тогда доля признаётся переоценкой тальк-подобной текстуры (гейт согласия)
        if m.talc_pct > thr and vote is not None:
            out.append({"factor": f"Доля талька {m.talc_pct:.1f}% > порога {thr:.0f}%, но "
                                  f"image-level модель не подтвердила оталькованность (p={vote:.0%}); "
                                  "доля отнесена к переоценке тальк-подобной текстуры",
                        "weight": "высокий"})
        else:
            out.append({"factor": f"Доля талька {m.talc_pct:.1f}% ≤ порога {thr:.0f}%",
                        "weight": "высокий"})
        if "fine_prob" in v.params:
            out.append({"factor": f"Вероятность тонких срастаний по модели: "
                                  f"{v.params['fine_prob']:.0%}", "weight": "решающий"})
        if feats.get("thick_med"):
            out.append({"factor": f"Медианная толщина сульфидных структур: "
                                  f"{feats['thick_med']:.1f} px (порог тонких ≈ 8)",
                        "weight": "высокий"})
        if feats.get("n_comp_per_ka"):
            out.append({"factor": f"Фрагментированность: {feats['n_comp_per_ka']:.1f} "
                                  f"комп./1000 px² сульфидов (порог ≈ 2.6)",
                        "weight": "средний"})
    out.append({"factor": f"Общая доля сульфидов: {m.sulfide_total_pct:.1f}%",
                "weight": "справочный"})
    return out


def _export_geojson(phase_map, to_src_scale: float, out_path: Path) -> None:
    """Полигоны фаз в координатах пикселей исходника."""
    import cv2

    from shlifscan.visualize import PH_FINE, PH_ORDINARY, PH_TALC

    names = {PH_ORDINARY: "ordinary", PH_FINE: "fine", PH_TALC: "talc"}
    features = []
    for ph, name in names.items():
        mask = (phase_map == ph).astype("uint8")
        if not mask.any():
            continue
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_TC89_L1)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1 or cv2.contourArea(cnt) < 30:
                continue
            eps = 1.5
            cnt = cv2.approxPolyDP(cnt, eps, True)
            if len(cnt) < 3:
                continue
            ring = (cnt.reshape(-1, 2) * to_src_scale).round(1).tolist()
            ring.append(ring[0])
            features.append({
                "type": "Feature",
                "properties": {"phase": name},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
    out_path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}), encoding="utf-8")
