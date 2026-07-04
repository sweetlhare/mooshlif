"""Экспертная логика классификации руды и итоговые метрики анализа."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


ORE_TALC = "оталькованная"
ORE_ORDINARY = "рядовая"
ORE_REFRACTORY = "труднообогатимая"


@dataclass
class OreMetrics:
    """Количественные метрики одного изображения/панорамы (доли в % площади шлифа)."""

    sulfide_total_pct: float = 0.0      # общая доля сульфидов
    ordinary_pct: float = 0.0           # сульфиды в обычных срастаниях
    fine_pct: float = 0.0               # сульфиды в тонких срастаниях
    talc_pct: float = 0.0               # доля талька
    gray_phase_pct: float = 0.0         # серая нерудная фаза (магнетит и пр.)
    analyzed_area_px: int = 0           # площадь проанализированной области, px исходника
    # доли типов срастаний от всех сульфидов, %
    ordinary_share: float = 0.0
    fine_share: float = 0.0
    # средняя уверенность моделей на спорных пикселях [0..1]
    confidence: float = 1.0


@dataclass
class OreVerdict:
    ore_class: str
    metrics: OreMetrics
    rationale: str = ""
    params: dict = field(default_factory=dict)


def load_intergrowth_model(models_dir: str | Path = MODELS_DIR):
    """Классификатор срастаний: ансамбль (морфология+DINOv2), если доступен
    его бэкбон, иначе — чистый GBM по морфологии. None, если файлов нет."""
    import joblib

    d = Path(models_dir)
    ens = d / "intergrowth_ensemble.pkl"
    if ens.exists():
        try:
            from .embeddings import _lazy_init

            if _lazy_init():
                return joblib.load(ens)
        except Exception:
            pass
    gbm = d / "intergrowth_gbm.pkl"
    if gbm.exists():
        return joblib.load(gbm)
    return None


def load_talc_vote(models_dir: str | Path = MODELS_DIR):
    """Image-level «голос» оталькованности по DINOv2-эмбеддингу (None если нет)."""
    p = Path(models_dir) / "talc_vote_lr.pkl"
    if not p.exists():
        return None
    import joblib

    return joblib.load(p)


def decide(
    metrics: OreMetrics,
    talc_thr_pct: float = 10.0,
    features: Optional[dict] = None,
    model_bundle: Optional[dict] = None,
    embedding=None,
    talc_vote: Optional[dict] = None,
    talc_vote_prob: Optional[float] = None,
    talc_frac_vote_gate: float = 0.5,
) -> OreVerdict:
    """Экспертное правило конкурса.

    1) измеренная доля талька > порога И это подтверждает image-level голос
       (>talc_frac_vote_gate) → оталькованная. Гейт голоса нужен потому, что
       пиксельная доля переоценивает тальк-подобную текстуру в диапазоне
       10–40% (артефакт калибровки): требуем согласия двух сигналов, иначе
       правило доли даёт много ложных срабатываний (валидировано: FP 46→19);
    1б) иначе image-level голос по DINOv2-эмбеддингу (тёмный домен съёмки
        физически скрывает текстуру талька от пиксельной модели — паттерн
        оталькованной руды ловится по виду снимка целиком);
    2) иначе тип срастаний: ансамбль (морфология + DINOv2) или доли площадей.
    """
    m = metrics
    params: dict = {"talc_thr_pct": talc_thr_pct}

    # --- диагностические сигналы считаем ВСЕГДА (для интерпретируемости и
    #     офлайн-переоптимизации правила); на решение влияют только пороги ниже ---
    vote_prob: Optional[float] = None
    if talc_vote is not None and (embedding is not None or talc_vote_prob is not None):
        try:
            vote_prob = (talc_vote_prob if talc_vote_prob is not None else float(
                talc_vote["model"].predict_proba(embedding[None])[0, 1]
            ))
            params["talc_vote_prob"] = round(vote_prob, 3)
        except Exception:
            vote_prob = None

    fine_prob: Optional[float] = None
    if model_bundle is not None and features is not None:
        try:
            import numpy as np

            x = np.array([[features[k] for k in model_bundle["features"]]])
            if model_bundle.get("kind") == "late_fusion":
                p_morph = float(model_bundle["morph_model"].predict_proba(x)[0, 1])
                emb = embedding if embedding is not None else None
                if emb is not None:
                    p_emb = float(
                        model_bundle["emb_model"].predict_proba(emb[None])[0, 1]
                    )
                    w = float(model_bundle.get("weight", 0.5))
                    fine_prob = w * p_morph + (1 - w) * p_emb
                else:
                    fine_prob = p_morph
            else:
                fine_prob = float(model_bundle["model"].predict_proba(x)[0, 1])
            params["intergrowth_model"] = model_bundle.get("version", "gbm")
            params["fine_prob"] = round(fine_prob, 3)
        except Exception:
            fine_prob = None

    # (1) правило доли талька с подтверждением голосом (если голос доступен)
    if m.talc_pct > talc_thr_pct and (vote_prob is None or vote_prob > talc_frac_vote_gate):
        cls = ORE_TALC
        confirm = (f" и подтверждено image-level моделью (p={vote_prob:.0%})"
                   if vote_prob is not None else "")
        rationale = (
            f"Содержание талька {m.talc_pct:.1f}% превышает порог "
            f"{talc_thr_pct:.0f}%{confirm}."
        )
        return OreVerdict(ore_class=cls, metrics=m, rationale=rationale, params=params)

    # (1б) image-level голос оталькованности
    if vote_prob is not None and vote_prob > float(talc_vote.get("thr", 0.6)):
        m.confidence = min(m.confidence, 0.5 + abs(vote_prob - 0.5))
        rationale = (
            f"Измеренная доля талька {m.talc_pct:.1f}% ≤ {talc_thr_pct:.0f}%, "
            f"но визуальный паттерн оталькованной руды распознан "
            f"image-level моделью (p={vote_prob:.0%})."
        )
        return OreVerdict(ore_class=ORE_TALC, metrics=m,
                          rationale=rationale, params=params)

    if fine_prob is not None:
        is_fine = fine_prob >= 0.5
        conf_note = f"вероятность тонких срастаний по модели {fine_prob:.0%}"
        m.confidence = min(m.confidence, 0.5 + abs(fine_prob - 0.5))
    else:
        is_fine = m.fine_share > m.ordinary_share
        conf_note = (
            f"доли по площади: тонкие {m.fine_share:.0f}% / обычные {m.ordinary_share:.0f}%"
        )

    if is_fine:
        cls = ORE_REFRACTORY
        rationale = (
            f"Тальк {m.talc_pct:.1f}% ≤ {talc_thr_pct:.0f}%; преобладают тонкие "
            f"срастания ({conf_note})."
        )
    else:
        cls = ORE_ORDINARY
        rationale = (
            f"Тальк {m.talc_pct:.1f}% ≤ {talc_thr_pct:.0f}%; преобладают обычные "
            f"срастания ({conf_note})."
        )
    return OreVerdict(ore_class=cls, metrics=m, rationale=rationale, params=params)


def conclusion_text(v: OreVerdict) -> str:
    """Краткое текстовое заключение для отчёта."""
    m = v.metrics
    parts = [
        f"Руда классифицирована как {v.ore_class}:",
        f"содержание талька — {m.talc_pct:.1f}%,",
        f"общая доля сульфидов — {m.sulfide_total_pct:.1f}%,",
        f"обычные срастания — {m.ordinary_share:.0f}% сульфидов,",
        f"тонкие срастания — {m.fine_share:.0f}% сульфидов.",
    ]
    return " ".join(parts) + " " + v.rationale
