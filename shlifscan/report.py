"""Экспорт результатов: таблица метрик (CSV), PDF-отчёт, лог параметров."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .classify import OreVerdict, conclusion_text


METRIC_COLUMNS = {
    "file": "Файл",
    "ore_class": "Класс руды",
    "sulfide_total_pct": "Сульфиды, всего %",
    "ordinary_pct": "Обычные срастания, % площади",
    "fine_pct": "Тонкие срастания, % площади",
    "ordinary_share": "Обычные, % сульфидов",
    "fine_share": "Тонкие, % сульфидов",
    "talc_pct": "Тальк, %",
    "confidence": "Уверенность",
}


def verdict_row(file: str, v: OreVerdict) -> dict:
    m = v.metrics
    return {
        "file": file,
        "ore_class": v.ore_class,
        "sulfide_total_pct": round(m.sulfide_total_pct, 2),
        "ordinary_pct": round(m.ordinary_pct, 2),
        "fine_pct": round(m.fine_pct, 2),
        "ordinary_share": round(m.ordinary_share, 1),
        "fine_share": round(m.fine_share, 1),
        "talc_pct": round(m.talc_pct, 2),
        "confidence": round(m.confidence, 3),
        "conclusion": conclusion_text(v),
    }


def save_csv(rows: Sequence[dict], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")  # BOM для Excel
    return df


def save_run_log(path: str | Path, config_json: str, files: Sequence[str],
                 extra: dict | None = None) -> None:
    """Лог параметров анализа для воспроизводимости."""
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": json.loads(config_json),
        "files": list(files),
    }
    if extra:
        log.update(extra)
    Path(path).write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_pdf_report(
    path: str | Path,
    file_name: str,
    verdict: OreVerdict,
    preview_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    legend_rgb: np.ndarray | None = None,
) -> None:
    """Одностраничный PDF-отчёт: исходник, маска, метрики, заключение.

    Использует matplotlib (кириллица поддерживается штатно).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = verdict.metrics
    fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
    fig.suptitle(
        f"ШЛИФ-Скан — отчёт по образцу: {file_name}", fontsize=14, fontweight="bold"
    )

    ax1 = fig.add_axes([0.03, 0.35, 0.45, 0.52])
    ax1.imshow(preview_rgb)
    ax1.set_title("Исходное изображение", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_axes([0.52, 0.35, 0.45, 0.52])
    ax2.imshow(overlay_rgb)
    ax2.set_title(
        "Сегментация: зелёный — обычные, красный — тонкие, синий — тальк",
        fontsize=10,
    )
    ax2.axis("off")

    ax3 = fig.add_axes([0.06, 0.06, 0.42, 0.24])
    ax3.axis("off")
    table_data = [
        ["Класс руды", verdict.ore_class.upper()],
        ["Сульфиды, всего", f"{m.sulfide_total_pct:.1f} %"],
        ["Обычные срастания (от сульфидов)", f"{m.ordinary_share:.0f} %"],
        ["Тонкие срастания (от сульфидов)", f"{m.fine_share:.0f} %"],
        ["Тальк", f"{m.talc_pct:.1f} %"],
        ["Уверенность анализа", f"{m.confidence:.2f}"],
    ]
    tbl = ax3.table(cellText=table_data, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    ax4 = fig.add_axes([0.52, 0.04, 0.45, 0.26])
    ax4.axis("off")
    import textwrap

    text = conclusion_text(verdict)
    ax4.text(0, 0.95, "Заключение:", fontsize=11, fontweight="bold", va="top")
    ax4.text(0, 0.8, "\n".join(textwrap.wrap(text, width=64)), fontsize=9, va="top")
    ax4.text(
        0, 0.02,
        f"Сформировано автоматически • {datetime.now():%d.%m.%Y %H:%M}",
        fontsize=7, color="gray", va="bottom",
    )

    fig.savefig(path, format="pdf")
    plt.close(fig)
