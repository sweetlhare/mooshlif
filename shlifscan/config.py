"""Конфигурация пайплайна: все пороги и параметры в одном месте.

Значения подобраны/валидированы на датасете конкурса (EDA: два домена
съёмки — «оливковый» ч1 и «тёмный» ч2/панорамы). Любой параметр можно
переопределить через JSON-конфиг; конфиг сохраняется рядом с результатами
анализа — это обеспечивает воспроизводимость.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessConfig:
    # длинная сторона рабочего изображения; панорамы тайлятся в полном разрешении
    analysis_max_side: int = 2048
    # перцентили stretch-нормализации яркости L (валидировано EDA)
    norm_percentiles: tuple[float, float] = (1.0, 99.7)
    # вырезать вшитую шкалу/подписи
    remove_scale_bar: bool = True


@dataclass
class SegmentConfig:
    # радиусы морфологического открытия (px): у серой фазы больше,
    # чтобы давить хроматические ореолы на границах ярких зёрен
    sulf_open_px: int = 3
    gray_open_px: int = 5


@dataclass
class IntergrowthConfig:
    """Пороги валидированы EDA (AUC 0.94–0.99 на 52 снимках обоих доменов).

    Признаки масштабируются к референсной ширине 2000 px.
    """

    ref_width: int = 2000
    # медианная толщина сульфидных структур (2×медиана distance transform), px:
    # ниже порога → тонкое срастание (медианы классов: ~13 vs ~5.5)
    thickness_thr_px: float = 8.0
    # фрагментированность: компонент на 1000 px² сульфидной площади
    n_comp_per_ka_thr: float = 2.6
    # радиус закрытия для группировки фрагментов в агрегаты
    aggregate_closing_px: int = 12
    # минимальная площадь агрегата для классификации, px
    min_aggregate_px: int = 64


_DEFAULT_TALC_MODEL = str(Path(__file__).resolve().parents[1] / "models" / "talc_unet.pt")


@dataclass
class TalcConfig:
    # путь к весам U-Net; если файла нет — классический детектор (fallback)
    model_path: Optional[str] = _DEFAULT_TALC_MODEL
    tile: int = 512
    overlap: int = 64
    prob_thr: float = 0.5
    # доля талька, выше которой руда оталькованная, %
    talc_ore_thr_pct: float = 10.0
    # правило доли (тальк>порога) срабатывает только при подтверждении
    # image-level голосом (>gate): пиксельная доля переоценивает тальк-подобную
    # текстуру в диапазоне 10–40%, поэтому требуем согласия двух сигналов
    talc_frac_vote_gate: float = 0.5


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    intergrowth: IntergrowthConfig = field(default_factory=IntergrowthConfig)
    talc: TalcConfig = field(default_factory=TalcConfig)
    # тайлинг панорам (px исходника)
    pano_tile: int = 4096
    pano_overlap: int = 256
    pano_min_side: int = 6000
    # даунскейл для расчёта глобальных констант нормализации панорамы
    pano_norm_side: int = 4096

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        sub_types = {
            "preprocess": PreprocessConfig,
            "segment": SegmentConfig,
            "intergrowth": IntergrowthConfig,
            "talc": TalcConfig,
        }
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in raw:
                continue
            v = raw[f.name]
            if f.name in sub_types:
                v = sub_types[f.name](**v)
            kwargs[f.name] = v
        return cls(**kwargs)
