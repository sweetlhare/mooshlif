[🇷🇺 Русская версия](README.ru.md)

# ShlifScan — automatic ore classification from polished-section panoramas

Solution for the **"Tell me who your section is"** track (Nornickel, 2026).

End-to-end system: a panoramic reflected-light (OM) image of a polished section
(up to gigapixel) → per-pixel phase segmentation → quantitative metrics → an
explained ore processing class → interactive viewing and reports.

## What the system does

1. **Segments phases** in reflected light: sulphides, grey non-ore phase
   (magnetite, etc.), talc, matrix. Works across two different imaging domains
   with no manual tuning (per-image adaptive LAB normalization).
2. **Classifies sulphide intergrowths**: coarse (green) vs fine (red) — from
   interpretable morphological features (structure thickness, fragmentation,
   replacement) + a DINOv2 ensemble.
3. **Estimates the talc fraction** (blue) with a trained model (U-Net on weak
   labels from expert outlines) with isotonic calibration of the fraction.
4. **Applies expert logic**: talc > 10% → *talc-bearing*; otherwise a
   predominance of fine intergrowths → *hard-to-process*, of coarse ones →
   *ordinary*.
5. **Produces the result**: a colour mask in a deep-zoom viewer, a metrics
   table, granulometry (P50/P80), a text conclusion, PDF/CSV/GeoJSON.

## Quick start (local)

```bash
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart segmentation-models-pytorch pyvips joblib

# CLI: analyze a single image or panorama
python -m shlifscan.cli analyze "data/Панорамы/4.jpg" -o reports/pano4

# CLI: batch-process a directory
python -m shlifscan.cli batch "data/Фото руд по сортам. ч2/рядовые" -o reports/batch

# Web application (backend + prebuilt frontend)
uvicorn app.backend.main:app --port 8000
# → http://localhost:8000
```

**Model weights.** Lightweight sklearn models (`models/*.pkl`) and calibrations
are kept in the repository. The heavy torch weights for the talc U-Net
(`models/talc_unet.pt`, ~98 MB) are not versioned in git — download them from
[GitHub Release v1.0](https://github.com/sweetlhare/mooshlif/releases/tag/v1.0)
(or from the solution archive on cloud storage) and place them in `models/`, or
retrain (`scripts/train_talc.py`). Without the `.pt`, the system runs on a
classic talc-detection fallback (`shlifscan/talc.py`, `_predict_classic`) — the
ore class stays correct, only the talc-fraction accuracy is lower.

**Frontend.** The prebuilt SPA (`app/frontend/dist/`) is included in the
repository — `uvicorn` serves it directly. To rebuild: `cd app/frontend && npm
ci && npm run build`. In the Docker image the frontend is built automatically
(multi-stage).

## Docker (on-prem)

```bash
docker compose up --build          # CPU profile
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build  # CUDA
```

Analysis artifacts and models are mounted as volumes (`runs/`, `models/`) — the
data never leaves the customer's perimeter.

## Architecture

```
image (TIFF/PNG/JPEG/BMP, up to 27000×21000)
   │  4096² tiling with overlap, global normalization constants
   ▼
[1] Preprocessing: LAB, percentile-stretch L, db = b − b_ref(matrix)
[2] Phase segmentation: adaptive rules (Otsu on db among bright pixels)
[3] Intergrowths: sulphide aggregates → morphological features →
    calibrated GBM (+ DINOv2 late fusion) → coarse/fine
[4] Talc: U-Net (resnet34, trained on weak labels from blue outlines
    with pCE+GCE+GatedCRF) → probabilities → calibrated fraction
[5] Expert logic + explanation → ore class
   ▼
DZI pyramids (pyvips) · PDF · CSV · GeoJSON · REST API · SSE progress
```

**Stack**: Python 3.10+, PyTorch (MPS/CUDA/CPU), OpenCV, scikit-image,
scikit-learn, segmentation-models-pytorch, FastAPI, pyvips, React 18 +
OpenSeadragon (deep zoom).

## Quality (validation on the competition data)

| Metric | Value | Protocol |
|---|---|---|
| Intergrowths (ordinary vs hard-to-process) | **macro-F1 0.915 ± 0.010** | 985 images, group-split 70/30 ×5, no leakage (MD5 dedup) |
| — morphology only (no DINOv2) | macro-F1 0.892 | same protocol |
| **Full 3-class task (talc agreement gate)** | **macro-F1 0.939 ± 0.012** | 5-fold GroupKFold by sample (section + MD5), 95% CI [0.924, 0.954]; the "talc > 10%" branch requires confirmation by an image-level vote |
| — same task before the gate (baseline) | macro-F1 0.905 | same protocol; the gate adds +0.034, talc-bearing precision 0.73→0.87, false positives 46→19 |
| Full 3-class task, early generalization estimate | macro-F1 0.888 ± 0.021 | 1100 images, group-split CV, thresholds tuned on the train fold |
| Agreement of the deployed system with the expert | acc **0.926** / macro-F1 0.905 | all 1180 images; ordinary F1 0.93, hard 0.95, talc-bearing 0.83 (recall 0.97) |
| — binary (ordinary vs hard-to-process) | acc 0.957 / F1 0.957 | 1051 images |
| Cross-domain part2→part1 / part1→part2 (intergrowths) | 0.941 / 0.877 | trained on one imaging domain, tested on the other |
| Talc-vs-rest by measured fraction (part1) | AUC 0.925 | 169 part1 images |
| External test: FeM (iron ore, Brazil) | IoU 0.87 / F1 0.93 / precision 0.99 | zero-shot, 20 fields, Zenodo 5014700 |
| External test: Cu ore (copper ore, Peru) | recall 0.988 / F1 0.86 (bright ore) | zero-shot, 22 fields, Zenodo 5020566 |
| Speed: 2272×1704 image | ~6 s (with DINOv2 and U-Net) | M2 Max (MPS) |
| Speed: 14999×10391 panorama (149 MP) | 32 s via the web API | target ≤ 5 min |

Talc-fraction estimation: a U-Net (encoder pretrained on LumenStone — polished
sections of Norilsk-group ores) on SAM-refined expert outlines + isotonic
calibration: val MAE 10.2 pp / IoU 0.47 **against the weak labels** (part of the
discrepancy is unlabelled talc outside the outlines). In the dark imaging domain
the talc texture is physically lost — in that case the ore class is determined
by the image-level model (talc-vote); the fraction's confidence interval is
honestly reported in the API/PDF.

A note on the data: the competition set contained 24 pairs of byte-identical
images with conflicting class labels (~4% noise) — they were excluded from
training; this caps the achievable ceiling of image-level metrics (~0.85
macro-F1 for 3 classes).

## Repository structure

```
shlifscan/          # core: the analysis pipeline (Python package)
  preprocess.py     #   normalization, artifact / scale-bar masks
  segment.py        #   phase segmentation (domain-adaptive rules)
  intergrowth.py    #   intergrowth features & classification, granulometry
  talc.py           #   talc detection (U-Net + classic fallback)
  classify.py       #   expert logic, ensemble, conclusion
  pipeline.py       #   orchestration, panorama tiling
  report.py         #   PDF/CSV, reproducibility log
  cli.py            #   command line (analyze / batch)
app/
  backend/          # FastAPI: analyses, SSE, DZI tiles, export
  frontend/         # React + OpenSeadragon SPA
scripts/            # training and validation
  train_talc.py     #   talc U-Net on weak labels
  validate_classification.py  # metrics on labelled folders
  extract_features.py         # intergrowth features
models/             # weights (torch .pt, sklearn .pkl) + manifests
docs/               # API contract, materials
```

## Metrological basis

Phase fractions are an unbiased estimate of the volumetric composition by the
Delesse principle (Aᴀ = Vᵥ, 1848; the Russian school — Glagolev's point method,
1933); automated image analysis in the spirit of ASTM E1245. The talc-fraction
confidence interval is two-component: between-field variance per ASTM E562
(t·s/√n_eff over a grid of fields) ⊕ the model's calibration error; the MSWD
test (Vermeesch 2018) flags spatial heterogeneity of talc. Russian framework:
GOST R ISO 9042-2011; the target NSOMMI / VIMS category — "quantitative
analysis" (S_repr < 30%).

## Reproducibility

- All thresholds/parameters live in `shlifscan/config.py`. The CLI path writes a
  full `run_log.json` (timestamp, the entire config, the file list) next to the
  result; the web path saves the same config to `runs/{id}/meta.json`, and model
  versions are served by `/api/health`.
- Model training: scripts in `scripts/` with fixed seeds and a group-split by
  sample (no train/val leakage).
- Talc labelling: masks are built from expert blue outlines (see
  `scripts/train_talc.py --prepare-only`); unreliable masks are excluded per a
  quality report.

## Fine-tuning (transfer learning)

The talc model can be fine-tuned on new data: label the regions (polygons in the
web UI — roadmap, or outlines in any editor), then:

```bash
python scripts/train_talc.py --masks <folder with masks> --epochs 25
```

The intergrowth classifier retrains in minutes:
`scripts/extract_features.py` → GBM (see the README in `scripts/`).
