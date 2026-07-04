# Дневник методов: что используем и откуда взято

Живой документ: каждый компонент решения → источник (статья / репозиторий /
индустриальная практика) → что именно взяли → где в коде. Обновляется при
каждом изменении пайплайна.

Легенда статуса: ✅ в проде · 🧪 эксперимент идёт · ❌ проверено и отклонено
(отрицательный результат зафиксирован).

## 1. Сегментация фаз (сульфиды / серая фаза / матрица)

| Компонент | Источник | Что взяли | Код | Статус |
|---|---|---|---|---|
| Попиксельные правила в LAB с адаптивным порогом жёлтости db = b − median(b\|тёмные) | собственная разработка (выбрана на EDA-сравнении против multi-Otsu и KMeans) | вся схема; Otsu по db среди ярких пикселей с защитами от вырожденных случаев | `shlifscan/segment.py` | ✅ |
| percentile-stretch нормализация яркости p1–p99.7 | стандартная CV-практика (robust contrast stretching) | нормализация перед порогами | `shlifscan/preprocess.py` | ✅ |
| medianBlur по каналу b против хроматических ореолов | собственная находка EDA (артефакт границ ярких зёрен) | подавление ложных «сульфидных» ободков (1.5%→0.13% FP) | `preprocess.py` | ✅ |
| Глобальные константы нормализации на панораму (не per-tile) | собственная находка EDA: per-tile нормализация даёт дрейф долей ×1.25 | расчёт констант по даунскейлу панорамы целиком | `pipeline.py:_pano_norm_constants` | ✅ |
| Multi-Otsu по L | Otsu 1979 / skimage | — | — | ❌ разваливается на панорамах (режет тёмную матрицу) |
| KMeans(L,a,b) + правила на центроидах | классика | — | — | ❌ качество ≈ правилам, но в 10–30× медленнее |
| Внешняя валидация правил | датасет FeM: Zenodo 5014700 (CC-BY 4.0, отражённый свет + SEM-маски, железная руда, PUC-Rio) | тест переносимости: IoU 0.87 / F1 0.93 / precision 0.99 без подстройки | `scratchpad/exp2/fem` | ✅ (число для презентации) |
| Внешняя валидация №2 | датасет Cu ore: Zenodo 5020566 (CC-BY 4.0, медная руда Перу, та же группа; бенчмарк Filippo 2021 DeepLabv3+ F1 0.921 — обученный на этих данных) | zero-shot: recall 0.988 / F1 0.858 по ярким рудным (vs Otsu-прокси); precision 0.982 на ore-vs-resin | `scratchpad/exp3_cu` | ✅ |
| Метрологическая рамка | Delesse 1848 → Глаголев 1933 → ASTM E562/E1245, ISO 9042:2024, ГОСТ Р ИСО 9042-2011; категории НСОММИ/ВИМС (количественный: Sвоспр<30%) | позиционирование метода + формулы CI | `stereology.py`, презентация | ✅ |

## 2. Классификация срастаний (обычные vs тонкие)

| Компонент | Источник | Что взяли | Код | Статус |
|---|---|---|---|---|
| Толщина структур: 2×медиана distance transform | классическая математическая морфология (Serra); аналог local thickness в металлографии | главный признак (AUC 0.976 на EDA-выборке) | `intergrowth.py` | ✅ |
| Удельный периметр perim/area | аналог PSIA (phase specific interfacial area) из mineral liberation analysis: Pérez-Barnuevo et al. 2013 (автоматическая классификация типов срастаний) | признак + идея «межфазная поверхность = тонкость срастания» | `intergrowth.py` | ✅ |
| Замещённость inclusion_frac (доля тёмного внутри closing) | идея из определения задачи («сульфиды, замещённые нерудной фазой») + морфологическое закрытие | признак AUC 0.948 | `intergrowth.py` | ✅ |
| Гранулометрия ECD P50/P80 (area-weighted) | стандарт отчётов автоматизированной минералогии (QEMSCAN/MLA deliverables; ASTM E1245-подобные метрики) | признаки + отчётная кривая для геологов | `intergrowth.py:granulometry_curve` | ✅ |
| Калиброванный GBM поверх признаков | sklearn HistGradientBoosting + CalibratedClassifierCV (isotonic; Zadrozny & Elkan 2002) | классификатор вердикта, holdout 0.893 | `models/intergrowth_gbm.pkl` | ✅ |
| DINOv2-эмбеддинги как второй голос (late fusion 0.5/0.5) | DINOv2: Oquab et al. 2023 (arXiv 2304.07193, Apache-2.0); прецедент few-shot в геологии: «DINOv2 Rocks» (arXiv 2407.18100) | frozen ViT-S/14, CLS⊕mean-patch, LogReg C=0.01; +2.3 п.п. → 0.915 | `shlifscan/embeddings.py`, `models/intergrowth_ensemble.pkl` | ✅ |
| Group-split по MD5 содержимого против утечек | общеизвестная практика; необходимость выявлена нами (24 пары дублей с конфликтными метками в датасете конкурса) | протокол всех валидаций | `scripts/*` | ✅ |

## 3. Детекция талька

| Компонент | Источник | Что взяли | Код | Статус |
|---|---|---|---|---|
| U-Net resnet34 | Ronneberger 2015; библиотека segmentation_models_pytorch (MIT) | базовая архитектура | `scripts/train_talc.py` | ✅ |
| Seed-карты из обводок: эрозия=позитив, кольцо/разрывы=ignore, вне обводок=ignore (PU-логика) | scribble-supervised практика: ScribbleSup (Lin 2016), выводы ScribbleBench (MICCAI 2025: простой pCE-бейзлайн обходит спец-методы вне их домена); PU-трактовка неисчерпывающей разметки | схема разметки слабых масок | `train_talc.py:prepare_examples` | ✅ |
| partial CE (ignore_index) | там же | лосс по размеченным пикселям | `train_talc.py` | ✅ |
| GCE-лосс q=0.7 на позитивах | Zhang & Sabuncu 2018 (arXiv 1805.07836, noise-robust loss) | устойчивость к шуму внутри обводок (сульфиды и пр.) | `train_talc.py` | ✅ |
| Gated CRF loss | Obukhov et al. 2019 (arXiv 1906.04651); референс-реализация WSL4MIS (HiLab-git) | компактная своя реализация (цвет+координаты, окно 11, даунскейл 4) | `train_talc.py:gated_crf_loss` | ✅ |
| Жёсткие цветовые аугментации > нормализации | Tellez et al. 2019 (arXiv 1902.06543, гистопатология) | HSV/brightness/gamma jitter в обучении | `train_talc.py:build_augmentation` | ✅ |
| Reinhard-перенос статистик LAB к референсу обучающей выборки | Reinhard et al. 2001 (Color Transfer between Images); практика stain normalization | доменная нормализация вход U-Net (оливковый↔тёмный) | `preprocess.py:reinhard_to_reference` | ✅ |
| Scale-jitter ±50% | выявленная диагностикой чувствительность к магнификации (5x/10x/20x, метаданных нет) | аугментация RandomScale | `train_talc.py` | ✅ |
| Мультискейл-инференс (1.0/0.7/0.5) | стандартный multi-scale TTA; выбран нашей диагностикой (−4 п.п. MAE) | усреднение вероятностей | `shlifscan/talc.py` | ✅ |
| Изотоническая калибровка ДОЛИ (raw_frac→true_frac) | isotonic regression (Zadrozny & Elkan); применение к area fraction — наша адаптация | часть модели, val MAE 13.9→9.3 (v1) | `scripts/calibrate_talc.py`, `models/talc_calibration.json` | ✅ |
| SAMRefiner-чистка обводок | SAMRefiner (ICLR 2025, arXiv 2502.06756) + SAM ViT-B (Kirillov 2023, Apache-2.0) | своя реализация ядра: distance-transform точки + negative в ярких фазах + bbox + mask-prompt; 37 «почти плотных» масок | `scratchpad/exp2/samrefined`; обучение с `--dense` | ✅ в v5 (см. выше) |
| petroscope/LumenStone энкодер-претрейн | Korshunov, Khvostikov et al., Mining Science and Technology 2025; github.com/xubiker/petroscope (LumenStone S2 = руды Норильской группы) | веса resnet34-энкодера ResUNet (216 тензоров) → инициализация нашей U-Net, decoder-first | `scripts/train_talc.py --init-encoder` | ✅ v5: вместе с SAM-масками MAE 12.7→10.2, IoU 0.41→0.47 · ⚠️ ЛИЦЕНЗИЯ petroscope = **GPL-3.0** (заразная), учесть при передаче заказчику |
| ConvNeXtV2-femto энкодер (ImageNet/FCMAE, permissive) как альтернатива petroscope | практика codemoo (эффективен в сегментации hand/pork); timm `tu-convnextv2_femto` в smp.Unet | обучен на тех же данных v5, тот же val-split; RTX 4090 | ❌ (проверено, обучение на сервере 188): raw crop val MAE 10.4% выглядел лучше, но продовый calib val MAE **19.25% vs v5 10.2%** — доменный претрейн petroscope бьёт ImageNet-архитектуру на слабой разметке/крошечном val; convnext компактнее (30 vs 98 МБ) и GPL-free → отложен как fallback для лицензионно-чистого варианта, требует лучшего рецепта/селекции эпохи |
| Self-training с классово-адаптивными порогами + rehearsal | упрощение UniMatch V2 (arXiv 2410.10777) и DARS (arXiv 2107.11279) | псевдоразметка ч2+панорам (pos≥0.75, neg≤0.12, ignore между) | `scripts/selftrain_talc.py` | ❌ в комбинации: отдельно 12.7→11.3 MAE, но v6 (self-train + SAM-маски + petroscope) = 18.8 MAE — конфликт псевдометок с плотной разметкой; прод = v5 без self-training |
| DINOv2-фичи + пиксельный LogReg (рецепт TLDR/Docherty) | arXiv 2410.19836 (сегментация чугуна в отражённом свете) | проверили как конкурента U-Net | `scratchpad/talc_dino` | ❌ проиграл (15.4 vs 9.3 п.п. MAE) |
| Talc-vote: image-level LogReg на DINOv2 | наша конструкция (мотив: De Castro & Benzaazoua 2022 — оптика не видит прозрачные минералы попиксельно; тёмный домен теряет текстуру) | второй сигнал решающего правила; для панорам — среднее per-tile вероятностей | `models/talc_vote_lr.pkl`, `classify.py` | ✅ |

## 4. Решающее правило и отчётность

| Компонент | Источник | Что взяли | Код | Статус |
|---|---|---|---|---|
| Порог «тальк > 10% → оталькованная» | ТЗ конкурса (экспертное правило заказчика) | сохранено дословно, порог настраиваемый | `classify.py:decide` | ✅ |
| Гейт согласия для правила доли (тальк>10% срабатывает только если image-level голос >0.5) | наша конструкция (мотив: пиксельная доля переоценивает тальк-подобную текстуру в 10–40% — плато калибровки; принцип согласия двух независимых сигналов) | подобрано честной 5-fold GroupKFold (группа=аншлиф, дубликаты по MD5); FP талька 46→19, оталькованная F1 0.833→0.912, macro-F1 0.905→0.939 (+0.037 на held-out); CV предпочитает вовсе игнорировать пиксельную долю (голос-only, 0.942), но гейт 0.5 оставлен ради сохранения ГОСТ-правила и участия квантификации в решении — разница 3 снимка, в пределах CV-шума ±0.011 | `classify.py:decide` (`talc_frac_vote_gate`), `scripts/optimize_rule.py` | ✅ |
| Отраслевой контекст порога | Mt Keith (BHP): операционный порог 2% талька, Talc Redesign +10% извлечения Ni (AusIMM MetPlant 2017); порог плавильни MgO<5% (911metallurgist) | аргументация ценности в презентации, не в коде | `docs/presentation` | ✅ |
| Доверительный интервал доли талька | ASTM E562-19e1 (межпольный CI = t·s/√n_eff) + принцип Delesse (Aᴀ=Vᵥ, 1848) + MSWD-тест неоднородности (Vermeesch 2018, EPSL 501) | двухкомпонентный 95% CI: пространственная (межпольная дисперсия по сетке полей) ⊕ модельная (MAE калибровки); флаг неоднородности | `shlifscan/stereology.py`, `worker.py` | ✅ |
| Индекс ассоциации сульфид↔тальк / сульфид↔серая | association index геометаллургии (Koch & Lund); мотивация: Chamlal & Benzaazoua 2025 (Minerals Eng 230:109406 — текстурный индекс предсказывает grade-recovery) | доля контактного периметра сульфидов с фазой; отчётная метрика обогатимости | `stereology.py:association_index` | ✅ |
| Широкий assay-CSV (одна строка = образец) | форматы импорта LIMS ГОКов: acQuire GIM Suite, Micromine Geobank, Datamine Fusion | плоские колонки с единицами в имени (TALC_PCT, ECD_P80_PX, ...) | `worker.py` (assay.csv) | ✅ |
| Текстовое заключение + панель «Объяснение решения» | требование ТЗ; формат факторов — наш | `classify.py:conclusion_text`, `worker.py:_explanation` | ✅ |

## 5. Продакшн-инженерия

| Компонент | Источник | Что взяли | Код | Статус |
|---|---|---|---|---|
| Deep Zoom (DZI) пирамиды | формат Microsoft Deep Zoom; генерация pyvips `dzsave` (libvips, LGPL) | тайлы 512+overlap 1, JPEG q90 / PNG-маски | `app/backend/dzi.py` | ✅ |
| Гигапиксельный вьюер | OpenSeadragon 4.x (BSD) | 2 DZI-слоя, opacity, navigator | `app/frontend` | ✅ |
| FastAPI + spawn-воркер + SSE-прогресс | стандартные паттерны (FastAPI docs; SSE для длинных задач) | очередь анализов, живой прогресс | `app/backend/main.py`, `worker.py` | ✅ |
| GeoJSON-экспорт масок | подход rasterio.features.shapes; реализовано на cv2.findContours (RETR_CCOMP) + approxPolyDP | ГИС-совместимость (QGIS/ArcGIS) | `worker.py:_export_geojson` | ✅ |
| Тайлинг с core-областями без швов | наша реализация; валидировано EDA (швы не дают артефактов) | `shlifscan/imio.py:iter_tiles` | ✅ |
| Docker CPU/GPU + офлайн-установка | практика on-prem поставки (аналоги: label-studio, CVAT) | multi-stage сборка, volume для моделей | `Dockerfile`, `docker-compose*.yml`, `install.sh` | ✅ |

## 6. Отклонено с обоснованием (чтобы не возвращаться)

- **Сырой SAM/SAM2 как semantic-сегментатор** — маски «протекают» на низком контрасте, нет классов (arXiv 2604.14805 + наш опыт).
- **Text-zero-shot (CLIPSeg/CAT-Seg/SAN)** — руды вне распределения CLIP.
- **Mask2Former / InternImage** — CUDA-кернелы (deformable attn / DCNv3), непрактично на MPS; при <100 масках не бьют U-Net (arXiv 2409.16940).
- **CAM/weakly-supervised из image-level меток** для масок срастаний — рвётся на тонких структурах.
- **STEGO/CutLER unsupervised** — валидированы только на natural images.
- **Собственный MAE/SSL-претрейн** — дороже и слабее готового DINOv2 при нашем бюджете времени.
- **DINOv3** — лицензия не Apache (регистрация, patent clauses) — в проде DINOv2. Apache-паритет отслеживаем: Franca (arXiv 2507.14137), Perception Encoder (arXiv 2504.13181).
- **SAM 3** — SAM License (не Apache), деривативы наследуют ограничения; SAM ViT-B (Apache) достаточно для чистки разметки.
- **Мультимодальные LLM как анализатор шлифов** — бенчмарк MatCha (arXiv 2509.09307): топ-MLLM значимо ниже эксперта на микроскопии; LLM уместен только как изложение результатов детерминированных моделей (роадмап).
- **KMeans/multi-Otsu сегментация** — см. §1.
- **fine_share по агрегатам как вердикт** — image-level признаки + GBM точнее (0.79→0.89).

### Bake-off робастных лоссов талька (04.07, сервер 188 / RTX 4090) — ❌ v5 сохранён
Гипотеза (из триажа findings): асимметричный шум метки (сульфид внутри контура помечен тальком) → AGCE (Zhou 2021) / T-Loss лучше GCE. Обучены с рецептом v5 (petroscope-init, freeze 5, 24 эпохи), тот же val.
| лосс | сырой crop val MAE | калиброванный VAL MAE (прод-пайплайн) |
|---|---|---|
| GCE q=0.7 (v5, прод) | 13.7% | **10.2%** |
| AGCE a=2 | 8.67% | 15.73% |
| AGCE a=4 | 12.85% | 15.02% |
| T-Loss (ν=2) | 10.59% | (прервано, сырой хуже a=2) |
Вывод: сырой crop-MAE обманчив (как у convnext), калиброванный хуже v5. Корень — крошечный шумный val (8 снимков): калибровка переобучается (train calib ~8% → val ~15%). **Бутылочное горло не лосс/бэкбон, а оценка.** Приоритет смещается на bootstrap-CI (сделано) и QuaPy-квантификацию (кандидат). Реализация лоссов: `scripts/train_talc.py --loss agce|tloss --agce-a`.

### QuaPy-квантификация доли талька (04.07) — ❌ изотоника сохранена
Кандидат №1 из триажа findings: доля талька = задача квантификации; ACC/PACC/SLD bias-корректируют ПЕРЕоценку классификатора. Реализованы напрямую (dependency-free), TPR/FPR по train-пикселям (метка eval_target), тот же val-таргет, что у изотоники.
| метод | VAL MAE |
|---|---|
| изотоника (прод) | **10.3%** |
| CC (сырой) | 15.1% |
| ACC | 17.4% |
| PACC | 18.2% |
| SLD/EMQ | 32.1% |
Вывод: квантификация проиграла. ACC/PACC предполагают стабильные TPR/FPR между доменами, но наша переоценка снимко-специфична → гибкая монотонная изотоника точнее; SLD/EMQ на 8 шумных val разносит. Подтверждает: бутылочное горло — крошечный шумный val + слабая разметка, а не метод оценки. `scripts/quantify_talc.py`.
