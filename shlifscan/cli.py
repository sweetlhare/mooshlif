"""CLI: анализ снимков/панорам, пакетный режим, экспорт отчётов.

Примеры:
    python -m shlifscan.cli analyze "data/Панорамы/4.jpg" -o reports/pano4
    python -m shlifscan.cli batch "data/Фото руд по сортам. ч2/рядовые" -o reports/batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .config import PipelineConfig
from .imio import SUPPORTED_EXT
from .pipeline import analyze_image
from .report import save_csv, save_pdf_report, save_run_log, verdict_row
from .visualize import confidence_heatmap, overlay


def _process_one(path: Path, out_dir: Path, cfg: PipelineConfig,
                 pdf: bool = True) -> dict:
    print(f"→ {path.name} ...", flush=True)
    res = analyze_image(path, cfg, progress=lambda p, msg: print(f"   {msg}", end="\r"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    ovl = overlay(res.rgb_preview, res.phase_map)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"),
                cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    heat = confidence_heatmap(res.rgb_preview, res.confidence)
    cv2.imwrite(str(out_dir / f"{stem}_confidence.jpg"),
                cv2.cvtColor(heat, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(out_dir / f"{stem}_phases.png"), res.phase_map)

    if pdf:
        save_pdf_report(out_dir / f"{stem}_report.pdf", path.name, res.verdict,
                        res.rgb_preview, ovl)

    row = verdict_row(str(path), res.verdict)
    row["elapsed_s"] = round(res.elapsed_s, 1)
    print(f"   {res.verdict.ore_class} | сульфиды {res.metrics.sulfide_total_pct:.1f}% | "
          f"тальк {res.metrics.talc_pct:.1f}% | {res.elapsed_s:.1f} c")
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="shlifscan", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="анализ одного снимка/панорамы")
    pa.add_argument("image")
    pa.add_argument("-o", "--out", default="reports/analyze")
    pa.add_argument("--config", help="JSON-конфиг пайплайна")
    pa.add_argument("--no-pdf", action="store_true")

    pb = sub.add_parser("batch", help="пакетная обработка директории")
    pb.add_argument("directory")
    pb.add_argument("-o", "--out", default="reports/batch")
    pb.add_argument("--config")
    pb.add_argument("--no-pdf", action="store_true")
    pb.add_argument("--limit", type=int, default=0)

    args = p.parse_args(argv)
    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "analyze":
        rows = [_process_one(Path(args.image), out_dir, cfg, pdf=not args.no_pdf)]
        files = [args.image]
    else:
        files = sorted(
            f for f in Path(args.directory).iterdir()
            if f.suffix.lower() in SUPPORTED_EXT
        )
        if args.limit:
            files = files[: args.limit]
        rows = []
        for f in files:
            try:
                rows.append(_process_one(f, out_dir, cfg, pdf=not args.no_pdf))
            except Exception as e:  # не роняем пакет из-за одного файла
                print(f"   ОШИБКА {f.name}: {e}", file=sys.stderr)
                rows.append({"file": str(f), "ore_class": "ошибка", "error": str(e)})
        files = [str(f) for f in files]

    save_csv(rows, out_dir / "metrics.csv")
    save_run_log(out_dir / "run_log.json", cfg.to_json(), files)
    print(f"Готово: {out_dir}/metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
