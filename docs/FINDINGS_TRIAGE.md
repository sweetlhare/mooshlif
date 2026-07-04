# Разбор всех находок для ШЛИФ

> Приземлено на реальность проекта (04.07, дедлайн сегодня): классификация 3-классовая теперь **vote-driven** (macro-F1 0.939), наша talc-модель **ПЕРЕсегментирует** (raw 0.48 vs true 0.25; 22–42% на рудах БЕЗ талька), val талька = 8 шумных снимков, self-training v6 дал регрессию 10.2→18.8 MAE, поставка Норникелю требует permissive-лицензий (petroscope=GPL-3.0 — риск).
>
> **Главный фильтр направления:** методы, толкающие к «больше позитивов» (PU/nnPU, pseudo-completion, scribble-expansion, entropy-min, focal/dice, small-target, dim-texture) для нас КОНТРпродуктивны. Нужно РАЗДЕЛЯТЬ истинный тальк и тальк-подобную серую текстуру.

---

## ЧАСТЬ 1. Сводные таблицы по ярусам вердиктов (методы/ресурсы, углы 1–7)

### 🟢 ADOPT-NOW — внедримо за часы без ретрейна (сегодня)

Отсортировано по ROI.

| Находка | Лицензия | Почему для нас | Next step |
|---|---|---|---|
| **Repeated/Nested CV + 0.632+ bootstrap** на GroupKFold-OOF | n/a (метод) | Прямо для headline vote-driven метрики (КРИТ №3): даёт CI на macro-F1 0.939 и F1 0.915 без ретрейна; лечит оптимистичное смещение малой выборки | Bootstrap-ресэмпл уже посчитанных OOF-предсказаний → CI на 0.939/0.915 в отчёт |
| **SegVal — perc. bootstrap + LOO CI** для сегментации | репо (метод тривиален, пишем сами) | Прямой ответ на КРИТ №4 (val=8): честно показывает ШИРОКИЙ CI; их таблица доказывает, что при n=8 узкий CI физически невозможен | Perc. bootstrap-CI + LOO для MAE доли талька (раздельно ч1/ч2) в отчёт/PROVENANCE |
| **TTA aleatoric uncertainty** (flip/rot90, БЕЗ яркости) | research | Без ретрейна (КРИТ №7): усредняет долю талька (↓дисперсия на шумном val), даёт карту неуверенности. Усредняет дисперсию, НЕ системный bias | flip/rot90-TTA на инференсе талька, усреднять вероятности до порога, отдавать pixel-std. Яркостные аугментации НЕ включать (усилят пере-сегментацию) |

### 🟡 TRY-IF-TIME — требует ретрейна/эксперимента (пост-дедлайн или при люфте)

Отсортировано по ROI и совместимости с нашим направлением ошибки.

| Находка | Лицензия | Почему для нас | Next step |
|---|---|---|---|
| **AGCE (ALFs)** — асимметричный GCE | MIT (verify) | Лучше всего матчит нашу реальность: шум асимметричный (label-talc-actually-sulfide). Near-drop-in апгрейд текущего GCE — самый дешёвый ретрейн | Первая, самая дешёвая арма bake-off: GCE→AGCE, тюнить asymmetry ratio, pCE+GatedCRF без изменений |
| **LAB-sulfide carve-out** (subtract sulfide from talc) | n/a (наши ассеты) | Единственный правильно-направленный ход угла 2: убирает FP-сульфиды ВНУТРИ контуров = меньше позитивов. Есть дешёвый no-retrain вариант | Пост-хок: вычесть LAB-sulfide маску из talc-предсказаний, ре-чек val MAE + false-talc rate. Оговорка: наш главный FP — серая ТЕКСТУРА, не сульфиды |
| **cleanlab — Confident Learning для сегментации** | Apache-2.0 | Offline → 0 риска обучающей петли (в отличие от v6). Ранжирует худшие из 8 val сегодня без ретрейна; локализует ошибки масок как комплемент SAMRefiner | CV → per-pixel pred_probs → find_label_issues → ранжировать худшие val/weak-label снимки (deployable сегодня) |
| **QuaPy — ACC/PACC, SLD/EMQ** (quantification) | BSD-3 | Наша доля талька = quantification; PACC bias-корректирует и ПЕРЕоценку (наш случай). Чистая лицензия | Прогнать PACC/SLD/EMQ поверх изотонической как sanity-check доли + CI; принять только если ↓MAE И ↓raw-долю на рудах без талька |
| **T-Loss** (Student-t NLL seg-loss) | Apache-2.0 | Авто-down-weight outlier-пикселей = supression FP; self-learned nu — плюс при 8-снимковом val | Арма pixel-loss bake-off вместо/в блендинге с pCE+GCE; судить по val MAE + FP-suppression на talc-free рудах |
| **ANL — Active Negative Loss** | CC-BY-4.0 | Asymmetric + imbalance = наш режим, где GCE слаб | Арма bake-off; ⚠️ следить за entropy-регуляризатором (может толкать к позитивам) — трекать raw-долю на talc-free |
| **GJS** (Generalized JS divergence) | MIT (verify) | Augmentation-consistency — БЕЗОПАСНЫЙ semi-sup сигнал (не хард-релейбл); CE↔MAE дил чище нашего GCE q | Арма bake-off; опционально consistency на unlabeled ч2 тайлах для домен-робастности |
| **ST++** (safe offline self-training) | MIT | Наименее рискованная правка существующей v6-инфры: обучаться только на стабильных между чекпойнтами снимках — обычно убирает регрессию | Перезапустить v6 с гейтом псевдоснимков по кросс-чекпойнт стабильности; валидировать против 10.2 MAE |
| **UniMatch V1** (weak-to-strong consistency) | MIT | Самый читаемый MIT-каркас портировать рецепт на НАШ U-Net; разделённые L_sup+L_unsup — лекарство от причины v6 | Форк dual-stream + confidence-threshold, вставить pCE+GCE+GatedCRF как L_sup на 8 масках каждый шаг (rehearsal) |
| **UniMatch V2** (DINOv2 weak-to-strong) | MIT | Мы уже на DINOv2; разделение лоссов + rehearsal | Пост-дедлайн student/EMA-teacher dual-stream на petroscope-энкодере, чекпойнт по калиброванной area-MAE |
| **SSL4MIS** (2D UNet Mean-Teacher/Cross-Teaching) | MIT | Ближайший практический код-старт к нашему стеку; MIT → и метод и файлы | Скопировать петли 2D-UNet Mean-Teacher/Cross-Teaching как стартовый каркас |
| **Cross Teaching CNN+Transformer** | MIT (SSL4MIS) | Практичный co-training: U-Net пиксель-голова + DINOv2-голова как гетерогенные учителя; дешевле CPS | 2D-UNet cross-teaching из SSL4MIS с нашими двумя головами |
| **CutMix-Seg** (color part) | MIT | Только цвет-часть (jitter/grayscale) сшивает ч1/ч2; пространственный CutMix опасен на границах фаз | При ретрейне: цвет-джиттер в strong-aug; осторожно с пространственным CutMix |
| **HR-Dv2** (DINOv2 upsampling + scribble logreg) | MIT | Reuse наш ViT-S/14; richer features могут РАЗДЕЛЯТЬ тальк от серой текстуры (КРИТ №1). Но INTERACTIVE | Offline на 1–2 ч2 панорамах: тест talc-vs-gray separation + чистка val масок; НЕ в vote-пайплайн сегодня |
| **vulture + interactive-seg-gui** (conv upsampler + XGBoost/RF) | MIT | RLM copper-ore-in-epoxy бенчмарк = НАША модальность; classical+DINOv2 features — путь к разделению | GUI на ч2 для relabel чистого val (КРИТ №4); тест снижения 22–42% false-talc |
| **Label refinement noisy scribble** (clean/noisy separation) | n/a | Убирает FP-сульфиды внутри контуров = убирает позитивы (правильно) | При ретрейне carve sulfide из seeds, GCE q=0.7 как база |
| **MicroNet (nasa/pretrained-microscopy-models)** | MIT | Единственный скачиваемый пермиссивный дроп-ин энкодер; кандидат заменить GPL-3.0 petroscope (КРИТ №6) | Загрузить resnet34/50 в SMP, переобучить U-Net тем же рецептом; критерий — сохраняет ли calib MAE ~10.2 при замене GPL→MIT |
| **MatSSL** (SSL-pretrain metallography encoder) | preprint (verify repo) | Безопасный путь эксплуатации unlabeled панорам (НЕ self-training, обходит v6); может заменить GPL-энкодер | Verify лицензию; SSL-адаптировать энкодер на unlabeled панорамах + IronOreRLM, fine-tune, сравнить calib MAE vs 10.2 |
| **FDA** (Fourier Domain Adaptation) | не указана (метод тривиален → портировать) | Мост ч1↔ч2 на unlabeled БЕЗ псевдометок (обходит v6); ⚠️ амплитуда трогает серую текстуру | Тест-тайм препроцессинг ч2→ч1, сравнить MAE vs Reinhard; принять только если НЕ раздувает raw-долю |
| **FACT** (Fourier augmentation DG) | research | Домен-робастность без адаптации под конкретный, безопаснее CycleGAN; та же каверза с амплитудой | Только при ретрейне, вместе с FDA; валидировать raw-долю на talc-free |
| **RandStainNA** (LAB stain-norm+aug) | research → портировать метод | LAB-вариант ложится на наш пайплайн; домен-робастность без GAN | Только при ретрейне: LAB-RandStainNA, ограниченный статистикой 2 доменов |
| **match_histograms** (skimage) | BSD-3 | Глубже Reinhard (вся форма распределения, не только mean/var) | Тест ч2→ референс ч1, замерить MAE по доменам; принять если не раздувает raw-долю |
| **AdaBN** (recompute BN stats on target) | n/a (метод) | Самый дешёвый адаптер ч1→ч2 на unlabeled, 0 обучаемых параметров; resnet34 имеет BN | Пересчитать BN running-stats на unlabeled ч2, замерить MAE по доменам; оставить если ↓пере-сегментацию |
| **Confident-learning (Lad&Mueller)** | AGPL-3.0 → портировать ИДЕЮ | Находит сульфиды в контурах + незамеченный тальк для чистки val; совместимо с направлением ошибки | Портировать CL-ранжирование (softmax vs метка); эксперт проверяет FP; cleanlab-пакет НЕ ставить (AGPL) |
| **DINOv2-embedding → talc% регрессор** | наш стек | Segmentation-free второй оценщик доли; ансамбль с U-Net ↓дисперсию (КРИТ №4), обходит прозрачность талька | Обучить регрессор на калиброванных долях, ансамбль с U-Net-долей, отчитать talc-% MAE/CI на GroupKFold |

### 🔵 PRETRAIN-ONLY — только для SSL-претрейна энкодера (пост-дедлайн)

| Находка | Лицензия | Почему для нас | Next step |
|---|---|---|---|
| **IronOreRLM** (563 RLM-снимка Fe-руды) | CC BY 4.0 (verify) | Единственный новый корпус В НАШЕЙ модальности (отражённый свет); SSL не добавляет позитивов → не контрпродуктивен (в отличие от v6). Но оксиды Fe, ноль масок, нет талька | Скачать в unlabeled пул; пост-дедлайн SSL continued-pretraining + Reinhard/домен стресс-тест. НЕ трогать labeled пайплайн сегодня |

### ⚪ REFERENCE — держать как справочник/дисциплину, не внедрять

| Находка | Лицензия | Почему reference | Что забрать |
|---|---|---|---|
| **Confident Learning MICCAI2020** (teacher-student + spatial smoothing) | not stated | Блюпринт correction-шага ЕСЛИ действуем на cleanlab error-map | Spatial-label-smoothing для SOFT корректировок вместо хард-катов |
| **DARS** (re-distribute biased pseudo labels) | Apache-2.0 | Distribution-alignment можно использовать как ПОТОЛОК (тянуть долю псевдо-талька ВНИЗ к prior) | Двунаправленная идея alignment, НЕ понижение порога |
| **DMT** (Dynamic Mutual Training) | BSD-3 | Ложится на наш ансамбль U-Net vs image-vote: down-weight там где расходятся | Понижать вес пикселей где U-Net='талёк', а vote<0.5 |
| **Class-driven Scribble Promotion** | unclear | Принципиальная версия нашего vote-гейта; vote-suppression на vote-negative = правильно | Валидирует, что shipped vote-gate — верный паттерн |
| **HELPNet (EGPL gating)** | unclear | EGPL «энтропия для GATE, не minimize» — здравый урок; SPR (closing outlines) — против нас | При ретрейне borrow только EGPL gating, skip SPR |
| **Mean Teacher / FixMatch / Arazo confirmation-bias** | n/a | Строительные блоки и диагностика ПОЧЕМУ v6 упал | EMA-учитель + rehearsal в любой будущий SSL-ретрейн; сослаться в PROVENANCE |
| **ELR/ELR+, SOP** | MIT | Нужен per-sample NN-loop, которого нет у LogReg/GBM голов (F1 0.915) | Reference, если голова станет epoch-trained NN |
| **CPS (TorchSemiSeg)** | MIT | Две полноценные сети — дороже Mean-Teacher за тот же выигрыш | Предпочесть Mean-Teacher/Cross-Teaching |
| **APL (NCE+RCE), SCE/SL** | MIT / research | Тот же design space что наш GCE, superseded ANL/ALFs | Baseline-anchor только если есть слот |
| **ShapePU (EM proportion)** | нет LICENSE → порт метода | EM-оценщик пропорций условно-нейтрален; per-phase доли для CI | Post-deadline: EM-оценщик vs изотоническая калибровка |
| **QuaPy prior-est (DEDPUL/TIcE/KM2)** | DEDPUL MIT | Оценка prior ill-posed на 8 шумных val; нужна была для nnPU (не берём) | DEDPUL (MIT) если понадобится независимая оценка доли |
| **PU FoV Consistency** | нет license | FoV-consistency против домен-шифта, но обёрнут в PU-лосс | Вынуть ТОЛЬКО crop-consistency БЕЗ PU-лосса при ретрейне |
| **ADELE, SP_guided, ScribbleVS, DMSPS** | not stated/unclear | Все корректируют «к уверенным предсказаниям»/расширяют seeds = наша ошибка/v6 | Borrow только benign JS-consistency (ADELE) / region-loss (SP_guided) / ignore-discipline |
| **Scribbles4All, ScribFormer, PLESS, OOD-SEG, 3D-BoxSup, Mask-the-Unknown** | mixed | Предполагают чистые dense маски / расширение / уже сделанное (ignore-кайма, КРИТ №2) | Reference/testbed на будущее; ignore-band уже реализует защиту |
| **FeatUp/LoftUp/AnyUp** | mixed (verify) | Enabling-блок для HR-Dv2/vulture; standalone нужен decoder retrain | FeatUp (permissive) как upsampling-блок если прототипим tldr-route |
| **convolt / COSE / Conformal-Risk / Sesia** | CC-BY/MIT/n/a | Conformal на 8 шумных снимках → покрытие ≤89%, бесполезно-широко; Conformal-Risk-на-FN против нашего направления | При десятках надёжных долей — калибровать; bootstrap-CI достаточно сейчас |
| **Quartz-in-resin Mask R-CNN (Ferreira 2024)** | n/a (нет кода) | Ближайший аналог: прозрачная фаза как регион с GT из коррелированной модальности (наш talc-vote) | В PROVENANCE как приоритет №1 пост-дедлайн итерации талька |
| **De Castro transparent-mineral (2021/2023), ltracegeo, GI/Copernicus, RoImAI, MME2025** | paywalled/none | Не та пробоподготовка/модальность или нет открытых весов; идеи переносимы, файлы нет | Framing в PROVENANCE; проверить Data Availability MME2025 |
| **YOLOv8n / PS-YOLO / Res-UNet+focal** | n/a | small-target/dim-texture/focal толкают к БОЛЬШЕ позитивов — против КРИТ №1 | Только для будущей instance-задачи по сульфидам, не тальк |

### 🔴 SKIP — не для нас (направление ошибки / лицензия / дедлайн)

| Находка | Лицензия | Почему skip |
|---|---|---|
| **PU / nnPU** (все варианты, все углы) | varies (ssnnpu=GPL-3.0) | ⛔ ГЛАВНЫЙ анти-паттерн: снимают штраф за FP вне контура → усиливают нашу ПЕРЕсегментацию (КРИТ №1). Премиса «недоразмечены» неверна |
| **T-HOneCls, aPU/imbalanced PU** | not stated/n/a | Тот же PU-вектор; «prior-free» лишь снимает зависимость от pi |
| **Tree Energy Loss** | Apache-2.0 | Propagate/fill sparse labels = больше coverage; нужен CUDA tree-filter op + ретрейн |
| **DSRG + AffinityNet (PSA)** | research-only/MIT | Seed-expansion/region-grow = больше позитивов; legacy Caffe/DeepLabv2 |
| **Bootstrapping (Reed), Taylor CE** | n/a (нет офиц. репо) | Наивное self-targeting = паттерн v6; тот же CE↔MAE дил что GCE |
| **Co-teaching/JoCoR, DivideMix** | MIT | Для image-level классификации с большими выборками; на 8 val small-loss selection нестабилен, выбросит minority тальк |
| **TENT** (test-time entropy-min) | MIT | Entropy-min как objective = больше позитивов (КРИТ №1); деградирует на малых батчах |
| **StainGAN/CycleGAN** | research | GAN не влезает в дедлайн; галлюцинирует тальк-подобную текстуру → усиливает пере-сегментацию |
| **Macenko/Vahadane** | repo | Физически неприменимы к отражённому свету (OD/Бер-Ламберт для проходящего H&E); у нас Reinhard-LAB |
| **AllSpark** | unverified | Тяжёлый трансформерный декодер, несовместим с нашим стеком/дедлайном |
| **Multimodal GNN EDS, ParticleSeg3D/BAM, Hyperspectral talc 3D-CNN** | varies/Apache/none | Не та модальность (SEM/EDS/µCT/гиперспектр); недеплоимо |
| **Menoufia MUMDMC2025, Roboflow BSE/thin-section** | CC BY 4.0 | Проходящий свет силикаты / BSE grayscale — не наша физика; нулевой трансфер |

---

## ЧАСТЬ 2. Научные находки по назначению (углы 8–15)

### 📖 CITE-IN-PRESENTATION — что и куда цитировать

| Находка (год) | Слайд / назначение | Ключевой тезис |
|---|---|---|
| **Korshunov / Хвостиков (LumenStone) 2021/2025** — DL-сегментация аншлифов, IoU 0.88, 9 фаз норильской ассоциации | Related-work / бейзлайн / провенанс энкодера | Ближайшая prior-art по Cu-Ni сульфидам; источник нашего petroscope resnet34. ⚠️ на слайд лицензий: petroscope=GPL-3.0 → портируем метод, не файлы |
| **徐述腾 & 周永章 2018** — U-Net отражённый свет ~91%, Acta Petrologica Sinica | Метод / related-work | Опубликованный прецедент подхода; «мало снимков → сильная аугментация». Честный контраст: у них плотный GT, у нас слабый надзор = труднее |
| **De Castro Dumont nickel 2023** (PhD, MOA vs QEMSCAN на Ni) | Intro / мотивация | Прямой прецедент «оптика → геометаллургический класс» на никеле, валидирован QEMSCAN — легитимирует наши 3 класса |
| **Lund 2015** (geometallurgical framework, текстурные архетипы) | Постановка задачи | «Ores of identical chemistry behave differently in processing solely due to texture» — ответ жюри «почему по картинке» |
| **Cropp 2013** (текстура + gangue → recovery, porphyry Cu review) | Обоснование классов | Два механизма (срастания → труднообогатимая; филлосиликаты/тальк → оталькованная) = наши две оси |
| **Nkomati 2013/2014** (pentlandite flotation, текстура) | Физика классов | Тот же Pn/Po/Cp: одинаковый grade → разное извлечение по текстуре/ассоциации; Po депрессирует Pn |
| **Pérez-Barnuevo 2013** (автоклассификация срастаний, отражённый свет) | Метод / intergrowth | Прямой прецедент нашей модальности; алгоритм объективнее эксперта; эмульсия = плохое раскрытие |
| **Chelgani talc flotation 2021 / Feng talc-serpentine 2012 / sodium phytate 2024** | Обоснование оталькованного класса + гейт >10% | Тальк природно гидрофобен (базальные силоксановые грани) → флотируется без коллектора, роняет качество Ni. Делает порог >10% металлургически осмысленным |
| **Хиден/Koch drill-core 2019, rapid-estimation 2021, Chamlal floatability 2025, Firdaus 2025** | Валидация парадигмы | Image texture → processability class — живой опубликованный подход; поддержка нашего image-level vote-branch |
| **Genkin-Distler 1981 / Изоитко 1997 / Атлас текстур 1958-64** | Вводный слайд / определение классов | Каркас доверия: терминология и таксономия из школы ИГЕМ, не выдумано под датасет |
| **Duryagina морфометрия 2022** (Октябрьское, Норильск) | Обоснование признаков (+ feature-idea) | Норильск-специфичный количественный якорь морфология-GBM |
| **Berrezueta Zaruma 2016 (CC-BY), Res-UNet ensemble 2024 (CC-BY)** | Метод / ансамбль | RGB-оптика (без мультиспектра) даёт количественную петрографию; ансамбль — установленный паттерн робастности |
| **composite particles 2023** (<25–50% exposed sulfide → collapse) | Механизм труднообогатимости | Твёрдое число вместо hand-wave: связывает тонкие срастания с провалом извлечения |
| **CN111282710B (Jinchuan), Algebraistova 2013, Durazzo 1982** | Признаки труднообогатимого класса | Тонкая вкрапленность + окисление + пламевидная эксолюция + магнетитовая рубашка. ЛИЦЕНЗИЯ: цитируем минералогию, не техпроцесс |
| **肖仪武 2018 (QEMSCAN/MLA словарь)** | Что различает рядовую/труднообогатимую | Язык: раскрытие / крупность вкрапленности / срастания / теор. извлечение — интерпретируемость для технологов |

### 💡 FEATURE-IDEA (только для ветки рядовая/труднообогатимая — её метрика НЕ vote-driven; try-if-time)

| Находка | Фича | Оговорка |
|---|---|---|
| **Duryagina 2022** | площадь, периметр, изрезанность края, сферичность, вытянутость, межзёрновое расстояние из LAB-сегментации → морфология-GBM | Дёшево (GBM refit, без ретрейна энкодера); но КРИТ №3 — слабо двигает 0.939 |
| **Lund AIM 2013** | Association Indicator Matrix (частоты контактов фаза-фаза) + «доля контактов Ni↔нерудное», «доля мелких контактов» | try-if-time; влияет на intergrowth-подмодель/assay, не headline |
| **Pérez-Barnuevo 2013** | contact/interface index, inclusion dispersion, inclusion size distribution → разделение простые/коарс vs эмульсия | Один из немногих рычагов, не заблокированных vote-метрикой (не-talc класс) |
| **µ-XRF GLCM k-means 2025** | GLCM-признаки (контраст/гомогенность/энергия/корреляция) на grayscale-патчах | ✅ Совпадает с КРИТ №1: может дискриминировать истинный тальк от серого шума. НЕ как objective «больше позитивов» — как пост-фильтр к U-Net-маске / в talc-vote |
| **王伟 2023 / 肖仪武 2018** | относительная гранулометрия сульфидных зёрен, прокси-раскрытие (доля одиночных vs сросшихся) | ⚠️ У нас НЕТ масштаба (подтверждено орг.) → только ОТНОСИТЕЛЬНАЯ крупность, мкм-пороги недоступны |
| **Iglesias 2018** | текстурная таксономия granular/lamellar/lobular + пористость | Их чёткость от циркулярной поляризации, которой у нас нет → считать с поправкой на шум |
| **YOLOv8n / Polished-section DL 2025** | small-target head для эмульсионных включений; superpixel/interactive labeling | ⛔ YOLOv8=AGPL-3.0 — порт концепции, не код. Post-deadline only |
| **Algebraistova 2013** | детекция магнетитовых кайм вокруг сульфидов (магнетит уже отдельная фаза в LAB) | try-if-time, НЕ к дедлайну |

### ⚠️ LIMITATION-TO-ACKNOWLEDGE — честные ограничения на слайд

| Находка | Ограничение | Как подать |
|---|---|---|
| **MSA talc optical properties 2022** | Отражательность талька ~5% vs сульфиды 35–50% → выглядит как смола/фон | КОЛИЧЕСТВЕННОЕ физ. объяснение, ПОЧЕМУ тальк пиксельно неотделим по яркости → опираемся на текстуру + image-vote |
| **Donskoi/De Castro AOM review 2022** | Прозрачные жильные силикаты (тальк, серпентин) плохо детектируются в отражённом свете | ГЛАВНАЯ цитата: published root-cause нашей пере-сегментации (КРИТ №1); оправдывает vote-driven и косвенную детекцию |
| **Pirard multispectral 2018/2004** | Pn/Po/Cp почти идентичны в single-band; фикс = мультиспектр (>99%) | У нас RGB → DINOv2-эмбеддинги как data-driven прокси дискриминативных бэндов; пре-эмптит вопрос о путанице фаз |
| **Becker NFG QEMSCAN 2009** | «Оталькованность» системна (тальк + сростки + entrainment > чистый тальк) | Наша пиксельная доля талька — шумный прокси, не абсолют; резонирует с переходом на vote-driven |
| **De Castro Part2 2023 / RLM-SEM Co-Site 2008 / Sandmann 2015** | Литературный фикс (акриловая пробоподготовка / SEM-ко-регистрация / MLA-эталон) нам НЕДОСТУПЕН | Объясняет шумность talc-GT (val=8) и почему val-MAE 10.2 трактовать осторожно; наш путь — алгоритмическое разделение, не съёмка |
| **Ueda seven texture indices 2017** | 2D-аншлиф систематически занижает 3D-раскрытие | Наши liberation/association-признаки — 2D-прокси, не абсолют |
| **AMCO multispectral 2020** | Эталон разделения сульфидов = мультиспектр, у нас RGB | future work: мультиспектральная съёмка → почти безошибочное разделение |
| **2021 Earth Science ML review** | Малые выборки + вариативность освещения/полировки + близкая отражательность — системные свойства модальности | Все три = ровно наши; наш ответ (Reinhard, vote-gate, изотон. калибровка) — инженерный ответ на признанное ограничение |
| **2018 Acta Petrologica (структуры распада 10–20 мкм)** | При нашем разрешении без шкалы напрямую не различаем | Ловим лишь макро-следствие (общую тонкость/пестроту текстуры) |
| **Русский Норильск-пробел** | НЕТ peer-reviewed статьи именно «Норильск оталькованная >10% → потери Ni» | Обоснование класса собираем из минералогии брекчий + флотохимии талька — честный limitation |

### 📚 BACKGROUND — фон/сноски, отдельного слайда не требуют

- **Depressors reviews** (中文 2020/2021 talc/serpentine depressants; polysaccharide talc 2015; сербентин-полиморфы 2026) — цитировать 1–2 представителя, остальное в список литературы.
- **Lund 2013/2014, ML-geometallurgy Peru 2025, Hilden 2017, Mishra Nkomati 2022, rapid-estimation** — методологический фон под GBM-ансамбль/многоминеральность.
- **Lopez-Benito 2020, Pirard 2004, multispectral sulfides 2019** — прецедент нашей LAB-цвето-сегментации.
- **Плаксинские чтения 2011-12, Лесникова 2018, bednyh nikelevyh rud** — русскоязычный контекст, grey literature (осторожно: без DOI/года).
- **De Castro PhD thesis 2023, DL-seg review 2023** — глубокие первоисточники для bibliography/appendix.
- **2019 fine mineral processing, MgO froth 1996** — замыкают цепочку «тонкая вкрапленность → и не раскрывается, и не флотируется».

---

## ЧАСТЬ 3. Полный разбор по каждому из 15 направлений

### Угол 1 — Label-noise-robust learning для talc U-Net
**Вердикт:** наполовину полезен, наполовину активно-неверен. USEFUL: пермиссивные drop-in robust-loss (AGCE первый, затем T-Loss, ANL, GJS) — все try-if-time на 1 ретрейне; cleanlab (Apache-2.0, offline) — безопасный быстрый диагност. WRONG: вся correction/relabeling/co-training ветвь (PU, ADELE, SP_guided, Bootstrapping, DivideMix, Co-teaching) клинчит с нашими фактами. Zero adopt-now; при люфте — GCE→AGCE + cleanlab-аудит.

| Находка | Лиц. | Вердикт |
|---|---|---|
| T-Loss | Apache-2.0 | try-if-time |
| cleanlab CL | Apache-2.0 | try-if-time |
| ANL | CC-BY-4.0 | try-if-time |
| GJS | MIT | try-if-time |
| AGCE (ALFs) | MIT | try-if-time |
| ADELE | not stated | reference |
| SP_guided | not stated | reference |
| APL (NCE+RCE) | MIT | reference |
| ELR/ELR+ | MIT | reference |
| SOP | MIT | reference |
| CL MICCAI2020 | not stated | reference |
| SCE/SL | research | reference |
| PU/nnPU | varies | **skip** |
| Taylor CE | n/a | skip |
| Bootstrapping (Reed) | n/a | skip |
| Co-teaching/JoCoR | MIT | skip |
| DivideMix | MIT | skip |

### Угол 2 — Weakly-supervised (scribbles/partial masks)
**Вердикт:** в основном мис-прицелен. Угол предполагает НЕДО-разметку, наша ошибка — ПЕРЕсегментация. Почти всё (DMSPS, TEL, HELPNet SPR, ScribbleVS, DSRG, PLESS, nnPU) толкает к БОЛЬШЕ позитивов. Правильно-направленные: (13) LAB-sulfide carve-out (try-if-time, no-retrain вариант), (10) vote-gated suppression (уже shipped). tldr-group (HR-Dv2/vulture, MIT) — offline для relabel val. Zero adopt-now.

| Находка | Лиц. | Вердикт |
|---|---|---|
| HR-Dv2 | MIT | try-if-time |
| vulture + GUI | MIT | try-if-time |
| Label refinement / LAB-sulfide carve-out | n/a | try-if-time |
| DMSPS | unclear | reference |
| HELPNet | unclear | reference |
| ScribbleVS | unclear | reference |
| Class-driven Scribble Promotion | unclear | reference |
| Scribbles4All | released | reference |
| ScribFormer | unclear | reference |
| FeatUp/LoftUp/AnyUp | mixed | reference |
| PLESS | n/a | reference |
| PU/nnPU | n/a | skip |
| Tree Energy Loss | Apache-2.0 | skip |
| DSRG + AffinityNet | research/MIT | skip |

### Угол 3 — PU-learning / partial-annotation
**Вердикт:** построен на ПЕРЕВЁРНУТОЙ посылке (что мы НЕДОдетектируем). nnPU и все PU снимают штраф за FP вне контура → усилят нашу ошибку. 0 adopt-now, 0 реальных try-if-time. Крупицы: (10) ignore-mask подтверждает уже сделанное + объясняет v6; (6) QuaPy (BSD-3) и (3) ShapePU EM — post-deadline sanity-check доли.

| Находка | Лиц. | Вердикт |
|---|---|---|
| QuaPy | BSD-3 | reference (→ try-if-time в угле 7) |
| ShapePU (EM) | нет license | reference |
| PU FoV Consistency | нет license | reference |
| Class-prior est (DEDPUL/TIcE) | varies | reference |
| OOD-SEG | CC-BY-4.0 | reference |
| 3D-BoxSup | n/a | reference |
| Mask-the-Unknown | CC-BY-4.0 | reference |
| Awesome-PU-learning | n/a | reference |
| nnPU | MIT | skip |
| ssnnpu | GPL-3.0 | skip |
| T-HOneCls | not stated | skip |
| aPU/imbalanced PU | n/a | skip |

### Угол 4 — Semi-supervised «правильно» (post-mortem v6)
**Вердикт:** диагноз v6 (confirmation bias + псевдометки топят 8 масок + стейл offline) КОРРЕКТЕН. Замена (online weak-to-strong с разделёнными лоссами) архитектурно здрава, лицензии permissive (UniMatch/SSL4MIS/ST++/CutMix/FreeMatch=MIT; DARS=Apache; DMT=BSD-3). Блокер — все требуют ретрейна. Threshold-lowering (FreeMatch SAT, DARS одностороннее) толкает к БОЛЬШЕ талька — reference-с-оговоркой. Лучшая ставка при люфте: ST++.

| Находка | Лиц. | Вердикт |
|---|---|---|
| ST++ | MIT | try-if-time |
| UniMatch V1 | MIT | try-if-time |
| UniMatch V2 | MIT | try-if-time |
| SSL4MIS | MIT | try-if-time |
| Cross Teaching CNN+Transformer | MIT | try-if-time |
| CutMix-Seg (color) | MIT | try-if-time |
| FixMatch | n/a | reference |
| Mean Teacher | n/a | reference |
| DARS | Apache-2.0 | reference |
| FreeMatch SAT | MIT | reference |
| CPS (TorchSemiSeg) | MIT | reference |
| DMT | BSD-3 | reference |
| Arazo confirmation-bias | n/a | reference |
| AllSpark | unverified | skip |

### Угол 5 — Свип открытого кода/весов (руды/минералы reflected light)
**Вердикт:** поле кодо-бедное, публичной модели талька в отражённом свете НЕ существует (подтверждено). Задействуема ОДНА: MicroNet (MIT, веса) — кандидат заменить GPL petroscope, try-if-time. Один фрейминг (reference): quartz-in-resin = talc-in-reflected-light. 0 adopt-now, 1 try-if-time, 1 pretrain-only (IronOreRLM), остальное reference/skip.

| Находка | Лиц. | Вердикт |
|---|---|---|
| MicroNet (nasa) | MIT | try-if-time |
| Quartz-in-resin Mask R-CNN | n/a | reference |
| YOLOv8n metal-mineral | n/a | reference |
| ltracegeo QEMSCAN U-Net | none | reference |
| PS-YOLO lithium | n/a | reference |
| Res-UNet+STN+dice/focal | n/a | reference |
| GI/Copernicus EDS-RF | open | reference |
| IronOreRLM | CC BY 4.0 | pretrain-only |
| Multimodal GNN EDS | varies | skip |
| ParticleSeg3D/BAM | Apache/custom | skip |
| Hyperspectral talc 3D-CNN | none | skip |
| Menoufia MUMDMC2025 | CC BY 4.0 | skip |

### Угол 6 — Open-data sweep (Zenodo/Mendeley/Kaggle/…)
**Вердикт:** near-total negative result = полезная находка: нового открытого labeled reflected-light Cu-Ni/talc корпуса НЕТ. 0 adopt-now. Ценность пост-дедлайн, пиксельная (заглушена vote-метрикой): IronOreRLM (SSL), MatSSL (безопаснее self-training), De Castro (talc-as-transparent reference).

| Находка | Лиц. | Вердикт |
|---|---|---|
| MatSSL | preprint (verify) | try-if-time |
| IronOreRLM | CC BY 4.0 | pretrain-only |
| De Castro transparent (2023) | paywalled | reference |
| RoImAI | не подтв. | reference |
| Polished Section DL (MME2025) | paywalled | reference |
| Roboflow BSE mineral seg | CC BY? | skip |
| Roboflow thin-section | CC BY 4.0 | skip |
| MUMDMC2025 | CC BY 4.0 | skip |

### Угол 7 — Надёжная оценка на крошечном шумном val + quantification + domain adaptation
**Вердикт:** хорошо прицелен в наши боли. 3 adopt-now: (1) bootstrap-CI/LOO (SegVal + repeated-CV/0.632+), (2) flip/rot-TTA. try-if-time: QuaPy PACC/SLD, портированный confident-learning. Режется реальностью: vote-driven, пере-сегментация (Conformal-Risk-на-FN, TENT контрпродуктивны), n=8 (conformal бесполезно-широк), дедлайн (CycleGAN/RandStainNA/FACT/FDA — только тест-тайм с проверкой на raw-долю). Macenko/Vahadane — физически skip.

| Находка | Лиц. | Вердикт |
|---|---|---|
| Repeated/Nested CV + 0.632+ bootstrap | n/a | **adopt-now** |
| SegVal bootstrap CI | репо | **adopt-now** |
| TTA aleatoric (no brightness) | research | **adopt-now** |
| QuaPy PACC/SLD | BSD-3 | try-if-time |
| Confident-learning (порт идеи) | AGPL→порт | try-if-time |
| FDA | не указана | try-if-time |
| FACT | research | try-if-time |
| RandStainNA | research | try-if-time |
| AdaBN | n/a | try-if-time |
| match_histograms | BSD-3 | try-if-time |
| convolt | CC-BY-4.0 | reference |
| Conformal Risk Control | MIT | reference |
| COSE | MIT | reference |
| Sesia adaptive conformal | n/a | reference |
| Label-Noise Overestimation | n/a | reference |
| TENT | MIT | skip |
| StainGAN/CycleGAN | research | skip |
| Macenko/Vahadane | repo | skip |

### Угол 8 — 中文 процессная минералогия Cu-Ni (Jinchuan/Hongqiling ≈ Норильск)
**Вердикт:** СИЛЬНЫЙ для презентации/ограничений, слабый для adopt-now. Топ-цитаты: #1 (徐述腾 U-Net, метод), #7 (肖仪武 язык обогатимости), #8 (Jinchuan патент, труднообогатимость), #5+#4 (тальк). Feature-идеи (относит. гранулометрия, прокси-раскрытие) — try-if-time, только рядовая/труднообогатимая. Мкм-пороги недоступны без масштаба. Лицензии чисты (литература/патент — цитируем метод/минералогию).

Назначения: 徐述腾2018 · 冯泽平2014 · 赵玉卿2018 · 肖仪武2018 · CN111282710B · Acta2018 = **cite**; 王伟2023 = **feature-idea**; Earth-Science2021 = **limitation**; депрессоры2020, fine-mineral2019 = **background**.

### Угол 9 — English process/technological mineralogy Cu-Ni
**Вердикт:** сильнейшее направление для defensibility. Peer-reviewed backbone «recovery контролируется ТЕКСТУРОЙ, не grade» (Mishra 2013, Nkomati 2022, Lund 2013/15), твёрдый механизм (<25–50% exposed → collapse), базис talc-класса (Feng 2012 + phytate 2024), Норильск-якорь Duryagina 2022. Donskoi 2022 + Pirard 2018 = published limits, объясняющие пере-сегментацию и vote-gate.

**cite:** Nkomati2013, Nkomati2022, Barnes2017, Lund2015, Dumont2023, Pirard(→limit), composite2023, Res-UNet2024, Korshunov, Feng2012, phytate2024. **feature-idea:** Duryagina2022, floatability2025. **limitation:** Donskoi2022, Pirard2018. **background:** Lund2013, Lopez-Benito2020.

### Угол 10 — Наука о тальке/серпентине в Cu-Ni
**Вердикт:** сильный, но по назначению — обоснование техклассов + честные ограничения, НЕ CV. Связки: «обоснование классов» (Chelgani2021 + Likhacheva2017 + Durazzo1982 + phytate2024 + MgO1996) и «оптическая невидимость талька» (MSA2022 + Donskoi2022 + Becker2009). Флотодепрессоры частично избыточны (1–2 представителя). Ни adopt-now, ни лицензионного риска.

**cite:** Likhacheva2017, Gipronikel2022, Chelgani2021, Edwards1980, Durazzo1982, phytate2024, MgO1996, Novel-optical2022. **limitation:** Becker2009, MSA2022, Donskoi2022. **background:** депрессоры2020/2021, polysaccharide2015, serpentine-polymorphs2026.

### Угол 11 — Компьютерный анализ руд в отражённом свете
**Вердикт:** ВЫСОКАЯ релевантность к объекту, но выхлоп — грунтовка презентации + ограничения. 3 опоры: De Castro2022 review (объясняет пере-сегментацию), Korshunov/Khvostikov (Cu-Ni + провенанс энкодера + GPL-флаг), De Castro Dumont (оптика→класс на никеле). Мультиспектр/пробоподготовка — в limitations/future-work.

**cite:** Korshunov2025, Khvostikov2021, Dumont2023, Berrezueta2016. **feature-idea:** Iglesias2018. **limitation:** De Castro review2022, De Castro Part2, AMCO2020, RLM-SEM2008. **background:** Pirard2004, multispectral2019, De Castro thesis2023.

### Угол 12 — Русскоязычная литература Норильск
**Вердикт:** сильный для презентации, слабый для инженерии. (1) каркас доверия (Генкин-Дистлер1981, Изоитко1997, Атлас1958-64); (2) количественный якорь Дурягина2022; (3) CV-прототип Коршунов2025. Пробел: НЕТ статьи «Норильск оталькованная >10%→потери Ni» → honest limitation. IoU 0.88 vs наш 0.47 — подавать как осознанный выбор (vote-driven).

**cite:** Дурягина2022, Коршунов2025, Алгебраистова2013, Генкин1981, Изоитко1997, Атлас1958-64. **background:** Лесникова2018, Плаксин2011-12.

### Угол 13 — Geometallurgy: texture→processing
**Вердикт:** очень ценный для защиты, не для метрик. 6 сильных цитат: мотивация (Lund2015, Cropp2013), физика (Nkomati2013, phytate2015), прецедент «классификация по картинке» в нашей модальности (Pérez-Barnuevo2013, Koch2019, Korshunov2025). 2 feature-идеи совпадают с болью: AIM (#2), GLCM-unsupervised (#13). 2 limitations: стереология 2D (Ueda2017), GPL petroscope.

**cite:** Lund2015, Cropp2013, Pérez-Barnuevo2013, Korshunov2025, Nkomati2013, Koch2019, polysaccharide2015. **feature-idea:** Lund-AIM2013, µXRF-GLCM2025. **limitation:** Ueda2017. **background:** Lund2014, Mishra2022, Hilden2017, rapid2021, ML-Peru2025, talc-serpentine2012.

### Угол 14 — Non-English + 2023-26 SOTA
**Вердикт:** сильнейший для нарратива, слабейший для нового кода. Ближайшая prior-art (De Castro2023 Dumont; Korshunov2025 LumenStone), металлургическое обоснование talc-класса (phytate2024, talc-serpentine2012, Nkomati2014), валидированная парадигма (rapid2021, Chamlal2025, Firdaus2025). 2 рычага: Pérez-Barnuevo эмульсия-дескрипторы (рядовая/труднообогатимая — не покрыто vote-gate); small-target heads (post-deadline). Caveats: vote-метрика заглушает пиксельный SOTA; YOLOv8=AGPL, petroscope/LumenStone=GPL — порт методов.

**cite:** De Castro2023, Korshunov2025, phytate2024, Mantilla2013, Filippo2021, Chamlal2025, Nkomati2014, rapid2021, Res-UNet2024, Firdaus2025. **feature-idea:** Pérez-Barnuevo2013, YOLOv8n2025. **limitation:** Sandmann2015, talc-serpentine2012. **background:** Pérez-Barnuevo2010, DL-seg-review2023.

---

## ЧАСТЬ 4. ИТОГ

### ✅ ТОП-5 ДЕЙСТВИЙ К ДЕДЛАЙНУ (adopt-now, часы, без ретрейна)

1. **Bootstrap-CI на headline-метрику.** Repeated-CV/0.632+ bootstrap на существующих GroupKFold-OOF → CI на macro-F1 0.939 и type F1 0.915. Честная неопределённость в отчёт/презентацию. (SegVal + repeated-CV, adopt-now)
2. **Bootstrap-CI + LOO на MAE доли талька** (раздельно ч1/ч2) на 8 val — прямо адресует КРИТ №4, показывает физически неизбежно широкий интервал при n=8.
3. **flip/rot90-TTA на инференсе талька** (БЕЗ яркостных аугментаций) — усредняет долю, даёт pixel-std карту неуверенности. Проверить долю на 8 val.
4. **LAB-sulfide carve-out (no-retrain)** — пост-хок вычесть LAB-sulfide маску из talc-предсказаний, ре-чек val MAE + false-talc rate. Единственный правильно-направленный пиксельный ход.
5. **Собрать презентацию-нарратив** на 6 опорных цитатах (Lund2015/Cropp2013 — texture→processability; Korshunov2025/De Castro-Dumont — прецедент; Chelgani2021/Feng2012/phytate2024 — talc-класс) + 3 limitation-слайда (MSA2022 отражательность 5%, Donskoi2022 прозрачные силикаты, Pirard Pn/Po/Cp), обращая слабости (IoU 0.47, пере-сегментация) в осознанные инженерные решения. Занести обоснования в docs/PROVENANCE.md.

**Если останется люфт (в порядке ROI):** GCE→AGCE bake-off arm · cleanlab offline-аудит 8 val · QuaPy PACC/SLD sanity-check доли.

### ⛔ ЧЕГО НЕ ДЕЛАТЬ И ПОЧЕМУ

- **PU / nnPU / все «treat-unlabeled-as-positive»** (ssnnpu, T-HOneCls, aPU) — ⛔ снимают штраф за FP вне контура → усиливают нашу ПЕРЕсегментацию (КРИТ №1). Наш вектор ПРОТИВОПОЛОЖНЫЙ: hard-negative mining из руд без талька.
- **Любой offline pseudo-completion / наивный self-training / Bootstrapping / DivideMix / Co-teaching** — паттерн, уже давший регрессию v6 (10.2→18.8, КРИТ №5). Если когда-либо — только ST++/online weak-to-strong с rehearsal и EMA-учителем.
- **Scribble-expansion / region-grow / outline-closing** (DMSPS, TEL, DSRG, HELPNet-SPR, ScribbleVS, PLESS) — все добавляют coverage = больше позитивов.
- **Entropy-min как objective** (TENT, ANL-entropy-часть, FreeMatch SAT threshold-lowering) — толкает к уверенным = больше позитивов; TENT ещё и деградирует на малых батчах.
- **focal/dice/small-target/dim-texture** (Res-UNet+focal, YOLOv8n, PS-YOLO) — up-weight редких/трудных позитивов = мимо (у нас ложные позитивы, а не пропуск).
- **CycleGAN/StainGAN** — не влезает в дедлайн, галлюцинирует тальк-подобную текстуру.
- **Conformal на 8 шумных val** (convolt/COSE/Conformal-Risk-на-FN) — покрытие ≤89%, бесполезно-широко; FN-контроль против нашего направления. Достаточно bootstrap-CI.
- **Macenko/Vahadane** — физически неприменимы к отражённому свету.
- **Любой ретрейн-зависимый метод сегодня** — КРИТ №7 (дедлайн), и КРИТ №3 (vote-driven) означает, что пиксельные улучшения талька почти не двигают 0.939 (двигают только assay-долю/CI).
- **Лицензии:** НЕ вендорить GPL/AGPL/кастом-файлы (petroscope=GPL-3.0, ssnnpu=GPL-3.0, cleanlab=AGPL-3.0, YOLOv8=AGPL-3.0, DINOv3-веса=Meta-restrictive, много repo без license) — портировать МЕТОД, не файлы. Пермиссивные предпочтительны (MIT/Apache/BSD/CC-BY): MicroNet, QuaPy, SSL4MIS, UniMatch, T-Loss, cleanlab-код (но пакет AGPL — порт идеи).