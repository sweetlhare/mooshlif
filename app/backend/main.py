"""ШЛИФ-Скан backend: FastAPI-приложение.

Запуск (dev):  uvicorn app.backend.main:app --reload --port 8000
Прод: см. docker/README. Артефакты анализов — в runs/ (volume).
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
DATA = ROOT / "data"
FRONTEND_DIST = ROOT / "app" / "frontend" / "dist"

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

app = FastAPI(title="ШЛИФ-Скан", version="1.0")
_executor: ProcessPoolExecutor | None = None


def executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(
            max_workers=1, mp_context=get_context("spawn")
        )
    return _executor


def _meta(run_id: str) -> dict:
    p = RUNS / run_id / "meta.json"
    if not p.exists():
        raise HTTPException(404, "анализ не найден")
    return json.loads(p.read_text(encoding="utf-8"))


def _feedback(run_id: str) -> dict:
    """Состояние обратной связи анализа (флаг «на доработку» + разметка)."""
    p = RUNS / run_id / "feedback.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"flagged": False, "reason": "", "note": "", "annotated": False}


def _write_feedback(run_id: str, fb: dict) -> None:
    (RUNS / run_id / "feedback.json").write_text(
        json.dumps(fb, ensure_ascii=False, indent=1), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _summary(meta: dict) -> dict:
    r = meta.get("result") or {}
    m = r.get("metrics") or {}
    fb = _feedback(meta["id"])
    return {
        "id": meta["id"],
        "file_name": meta["file_name"],
        "status": meta["status"],
        "created_at": meta["created_at"],
        "ore_class": r.get("ore_class"),
        "talc_pct": m.get("talc_pct"),
        "sulfide_total_pct": m.get("sulfide_total_pct"),
        "elapsed_s": meta.get("elapsed_s"),
        "error": meta.get("error"),
        "flagged": fb.get("flagged", False),
        "annotated": fb.get("annotated", False),
    }


# ------------------------------------------------------------------ анализы
@app.post("/api/analyses", status_code=202)
async def create_analysis(
    file: UploadFile | None = File(None),
    server_path: str | None = Form(None),
    params: str | None = Form(None),
):
    from datetime import datetime

    if file is None and not server_path:
        raise HTTPException(400, "нужен file или server_path")

    run_id = uuid.uuid4().hex[:8]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)

    if file is not None:
        ext = Path(file.filename or "image.jpg").suffix.lower()
        if ext not in SUPPORTED_EXT:
            shutil.rmtree(run_dir)
            raise HTTPException(400, f"формат {ext} не поддерживается")
        src = run_dir / f"original{ext}"
        with open(src, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        file_name = file.filename
    else:
        src = Path(server_path)
        if not src.is_absolute():
            src = ROOT / src
        # защита от выхода за пределы проекта
        try:
            src.resolve().relative_to(ROOT.resolve())
        except ValueError:
            raise HTTPException(400, "server_path вне каталога проекта")
        if not src.exists() or src.suffix.lower() not in SUPPORTED_EXT:
            shutil.rmtree(run_dir)
            raise HTTPException(400, "файл не найден или формат не поддерживается")
        file_name = src.name

    if params:
        try:
            json.loads(params)
        except json.JSONDecodeError:
            shutil.rmtree(run_dir)
            raise HTTPException(400, "params: некорректный JSON")

    meta = {
        "id": run_id,
        "file_name": file_name,
        "src_path": str(src),
        "status": "queued",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "params": params or "",
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    from app.backend.worker import run_analysis

    executor().submit(run_analysis, run_id, str(RUNS), str(src), params or "")
    return {"id": run_id, "status": "queued"}


@app.get("/api/analyses")
def list_analyses():
    out = []
    for d in sorted(RUNS.iterdir(), reverse=True):
        mp = d / "meta.json"
        if mp.exists():
            try:
                out.append(_summary(json.loads(mp.read_text(encoding="utf-8"))))
            except json.JSONDecodeError:
                continue
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out


@app.get("/api/analyses/{run_id}")
def get_analysis(run_id: str):
    meta = _meta(run_id)
    meta.pop("src_path", None)
    return meta


@app.delete("/api/analyses/{run_id}")
def delete_analysis(run_id: str):
    d = RUNS / run_id
    if not d.exists():
        raise HTTPException(404)
    shutil.rmtree(d)
    return {"ok": True}


@app.get("/api/analyses/{run_id}/events")
async def analysis_events(run_id: str):
    run_dir = RUNS / run_id

    async def stream():
        last = None
        while True:
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            prog_p = run_dir / "progress.json"
            prog = (json.loads(prog_p.read_text(encoding="utf-8"))
                    if prog_p.exists() else {"stage": "в очереди", "percent": 0})
            if meta["status"] == "done":
                yield f'data: {{"stage": "done", "percent": 100}}\n\n'
                return
            if meta["status"] == "error":
                err = json.dumps(meta.get("error", ""), ensure_ascii=False)
                yield f'data: {{"stage": "error", "percent": 100, "error": {err}}}\n\n'
                return
            cur = json.dumps(prog, ensure_ascii=False)
            if cur != last:
                yield f"data: {cur}\n\n"
                last = cur
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------- файлы
def _file(run_id: str, rel: str, media: str | None = None) -> FileResponse:
    p = RUNS / run_id / rel
    if not p.exists():
        raise HTTPException(404, f"{rel} не найден")
    return FileResponse(p, media_type=media)


@app.get("/api/analyses/{run_id}/dzi/{layer}.dzi")
def dzi_manifest(run_id: str, layer: str):
    return _file(run_id, f"dzi/{layer}.dzi", "application/xml")


@app.get("/api/analyses/{run_id}/dzi/{layer}_files/{level}/{tile}")
def dzi_tile(run_id: str, layer: str, level: int, tile: str):
    if "/" in tile or ".." in tile or ".." in layer:
        raise HTTPException(400)
    return _file(run_id, f"dzi/{layer}_files/{level}/{tile}")


@app.get("/api/analyses/{run_id}/preview.jpg")
def preview(run_id: str):
    return _file(run_id, "preview.jpg", "image/jpeg")


@app.get("/api/analyses/{run_id}/confidence.jpg")
def confidence(run_id: str):
    return _file(run_id, "confidence.jpg", "image/jpeg")


@app.get("/api/analyses/{run_id}/report.pdf")
def report_pdf(run_id: str):
    return _file(run_id, "report.pdf", "application/pdf")


@app.get("/api/analyses/{run_id}/metrics.csv")
def metrics_csv(run_id: str):
    return _file(run_id, "metrics.csv", "text/csv")


@app.get("/api/analyses/{run_id}/assay.csv")
def assay_csv(run_id: str):
    """Широкий assay-профиль (одна строка = образец) для импорта в LIMS."""
    return _file(run_id, "assay.csv", "text/csv")


@app.get("/api/analyses/{run_id}/mask.geojson")
def mask_geojson(run_id: str):
    return _file(run_id, "mask.geojson", "application/geo+json")


@app.get("/api/export/batch.csv")
def batch_csv():
    import io

    import pandas as pd

    rows = [
        _summary(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        for d in RUNS.iterdir() if (d / "meta.json").exists()
    ]
    buf = io.StringIO()
    pd.DataFrame([r for r in rows if r["status"] == "done"]).to_csv(buf, index=False)
    return Response(buf.getvalue().encode("utf-8-sig"), media_type="text/csv")


# --------------------------------------------------------------- служебные
@app.get("/api/health")
def health():
    import torch

    models = {}
    mdir = ROOT / "models"
    if (mdir / "talc_unet.pt").exists():
        ver = "v3"
        calib = mdir / "talc_calibration.json"
        if calib.exists():
            ver = json.loads(calib.read_text(encoding="utf-8")).get("model", ver)
        models["talc_unet"] = ver.replace("talc_unet.pt ", "").strip("()") or "v3"
    if (mdir / "intergrowth_ensemble.pkl").exists():
        models["intergrowth"] = "ensemble v1 (морфология+DINOv2)"
    elif (mdir / "intergrowth_gbm.pkl").exists():
        models["intergrowth"] = "gbm v2"
    if (mdir / "talc_vote_lr.pkl").exists():
        models["talc_vote"] = "v1"
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    return {"status": "ok", "models": models, "device": device}


@app.get("/api/demo-images")
def demo_images():
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    out = []
    pano_dir = DATA / "Панорамы"
    if pano_dir.exists():
        for f in sorted(pano_dir.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXT:
                with Image.open(f) as im:
                    out.append({
                        "name": f"Панорама {f.stem}",
                        "server_path": str(f.relative_to(ROOT)),
                        "size": list(im.size),
                    })
    return out


# ------------------------------------------------ обратная связь / дообучение

# фазы для разметки (под нашу задачу; цвета совпадают с контрактом маски)
ANNOTATION_PHASES = [
    {"key": "talc", "label": "Тальк", "color": "#2f6fd6"},
    {"key": "sulfide", "label": "Сульфиды", "color": "#e0a63b"},
    {"key": "gray", "label": "Серая нерудная фаза", "color": "#6a7484"},
    {"key": "fine", "label": "Тонкие срастания", "color": "#dc3a32"},
    {"key": "ordinary", "label": "Обычные срастания", "color": "#12995a"},
    {"key": "background", "label": "Фон / смола", "color": "#efe6d8"},
]


@app.post("/api/analyses/{run_id}/flag")
def flag_analysis(run_id: str, body: dict = Body(default={})):
    """Пометить анализ «на доработку» (плохой снимок → в очередь разметки)."""
    _meta(run_id)  # 404 если нет
    fb = _feedback(run_id)
    fb.update({
        "flagged": True,
        "reason": str(body.get("reason", ""))[:120],
        "note": str(body.get("note", ""))[:2000],
        "flagged_at": _now(),
    })
    _write_feedback(run_id, fb)
    return {"ok": True, "feedback": fb}


@app.delete("/api/analyses/{run_id}/flag")
def unflag_analysis(run_id: str):
    """Снять флаг «на доработку»."""
    _meta(run_id)
    fb = _feedback(run_id)
    fb["flagged"] = False
    _write_feedback(run_id, fb)
    return {"ok": True, "feedback": fb}


@app.get("/api/analyses/{run_id}/feedback")
def get_feedback(run_id: str):
    _meta(run_id)
    return _feedback(run_id)


@app.get("/api/feedback")
def list_feedback():
    """Очередь снимков на доработку (флаг «на доработку»), новые сверху."""
    items = []
    for d in sorted(RUNS.iterdir(), reverse=True):
        if not (d / "meta.json").exists():
            continue
        fb = _feedback(d.name)
        if fb.get("flagged"):
            s = _summary(_meta(d.name))
            s["reason"] = fb.get("reason", "")
            s["note"] = fb.get("note", "")
            s["flagged_at"] = fb.get("flagged_at")
            items.append(s)
    return items


@app.get("/api/annotation/phases")
def annotation_phases():
    """Палитра фаз для разметки под нашу задачу."""
    return ANNOTATION_PHASES


@app.post("/api/analyses/{run_id}/annotation")
def save_annotation(run_id: str, body: dict = Body(...)):
    """Сохранить разметку фаз (PNG-маска data-URL + легенда) для дообучения."""
    _meta(run_id)
    data_url = body.get("image", "")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
    except Exception:
        raise HTTPException(400, "некорректный PNG data-URL")
    (RUNS / run_id / "annotation.png").write_bytes(raw)
    (RUNS / run_id / "annotation.json").write_text(json.dumps({
        "phases": body.get("phases", ANNOTATION_PHASES),
        "width": body.get("width"),
        "height": body.get("height"),
        "saved_at": _now(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    fb = _feedback(run_id)
    fb.update({"annotated": True, "annotated_at": _now()})
    _write_feedback(run_id, fb)
    return {"ok": True}


@app.get("/api/analyses/{run_id}/annotation.png")
def get_annotation(run_id: str):
    p = RUNS / run_id / "annotation.png"
    if not p.exists():
        raise HTTPException(404, "разметки нет")
    return FileResponse(p, media_type="image/png")


# фронтенд-статика (после сборки)
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="spa")
