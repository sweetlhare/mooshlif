# ШЛИФ-Скан — API-контракт (backend ↔ frontend)

Backend: FastAPI, префикс `/api`. Все ответы JSON (кроме тайлов/файлов).
Статика фронтенда раздаётся с корня `/`.

## Анализы

### POST /api/analyses
Создать анализ. `multipart/form-data`:
- `file`: изображение (TIFF/PNG/JPEG/BMP, до гигапикселя) — ИЛИ
- `server_path`: строка, путь к файлу на сервере (для демо со встроенными панорамами)
- `params` (опц.): JSON-строка переопределений конфига (например `{"talc": {"talc_ore_thr_pct": 10}}`)

Ответ 202: `{"id": "a3f9c2", "status": "queued"}`

### GET /api/analyses
Список: `[{id, file_name, status, created_at, ore_class, talc_pct, sulfide_total_pct, elapsed_s}]`
`status ∈ {queued, running, done, error}`

### GET /api/analyses/{id}
Детали:
```json
{
  "id": "a3f9c2", "file_name": "4.jpg", "status": "done",
  "created_at": "2026-07-03T02:00:00", "elapsed_s": 141.2,
  "error": null,
  "image": {"width": 13712, "height": 10798, "work_scale": 0.5},
  "result": {
    "ore_class": "оталькованная",
    "conclusion": "Руда классифицирована как ...",
    "confidence": 0.87,
    "metrics": {
      "sulfide_total_pct": 12.3, "ordinary_pct": 8.1, "fine_pct": 4.2,
      "ordinary_share": 65.8, "fine_share": 34.2,
      "talc_pct": 14.0, "talc_ci": [12.1, 15.9],
      "gray_phase_pct": 9.5
    },
    "granulometry": {"ecd_p50": 38.2, "ecd_p80": 96.0, "curve": [[d, cum_frac], ...]},
    "features": {"thick_med": 5.6, "n_comp_per_ka": 3.1, "perim_per_area": 0.21,
                  "inclusion_frac": 0.44},
    "explanation": [
      {"factor": "Доля талька 14.0% > 10%", "weight": "решающий"},
      {"factor": "Медианная толщина сульфидов 5.6 px < 8", "weight": "высокий"}
    ],
    "params": {"talc_ore_thr_pct": 10.0, "model_version": "talc_unet_v1"}
  }
}
```

### GET /api/analyses/{id}/events  (SSE)
События прогресса: `data: {"stage": "сегментация", "percent": 42}` каждые ~1с;
финальное `data: {"stage": "done", "percent": 100}`. При статусе done/error поток
сразу шлёт финальное событие и закрывается.

### DELETE /api/analyses/{id}

## Тайлы (Deep Zoom, OpenSeadragon)

### GET /api/analyses/{id}/dzi/{layer}.dzi
XML-манифест DZI. `layer ∈ {image, phases}`:
- `image` — исходник (JPEG-тайлы)
- `phases` — цветная маска фаз (PNG-тайлы с альфой: зелёный=обычные срастания,
  красный=тонкие, синий=тальк, прозрачный=прочее)

### GET /api/analyses/{id}/dzi/{layer}_files/{level}/{col}_{row}.{ext}
Тайлы. `ext`: jpg для image, png для phases.

### GET /api/analyses/{id}/preview.jpg
Превью исходника ≤1600px (для списка/карточек).

### GET /api/analyses/{id}/confidence.jpg
Карта уверенности (heatmap, рабочий масштаб ≤2048px).

## Экспорт

- `GET /api/analyses/{id}/report.pdf` — PDF-отчёт
- `GET /api/analyses/{id}/metrics.csv` — метрики одной строкой
- `GET /api/analyses/{id}/mask.geojson` — полигоны фаз (properties.phase ∈
  {ordinary, fine, talc}, координаты в пикселях исходника)
- `GET /api/export/batch.csv` — сводный CSV по всем анализам

## Служебные

- `GET /api/health` → `{"status": "ok", "models": {"talc_unet": "v1"}, "device": "mps"}`
- `GET /api/demo-images` → список встроенных демо-файлов
  `[{"name": "Панорама 4", "server_path": "...", "size": [13712, 10798]}]`
