import { useCallback, useEffect, useState, type CSSProperties, type ReactNode } from 'react'
import { api, errorMessage, urls } from '../api/client'
import type { AnalysisDetail, AnalysisResult } from '../api/types'
import { useProgress } from '../hooks/useProgress'
import {
  FEATURE_LABELS,
  fmtDateTime,
  fmtElapsed,
  fmtNum,
  fmtPct,
  oreClassColor,
  STATUS_COLOR,
  STATUS_LABEL,
  weightColor,
} from '../lib/format'
import { OsdViewer, type LayerState } from '../components/OsdViewer'
import { EcdChart } from '../components/EcdChart'
import { Led, Modal, ProgressBar, Spinner } from '../components/ui'
import s from './AnalysisPage.module.css'

/* ================= Панель слоёв ================= */

function Switch({ on, onToggle, label }: { on: boolean; onToggle: () => void; label: string }) {
  return (
    <button
      className={`${s.switch} ${on ? s.switchOn : ''}`}
      onClick={onToggle}
      role="switch"
      aria-checked={on}
      aria-label={label}
      type="button"
    />
  )
}

const LEGEND = [
  { color: 'var(--phase-ordinary)', label: 'Обычные срастания' },
  { color: 'var(--phase-fine)', label: 'Тонкие срастания' },
  { color: 'var(--phase-talc)', label: 'Тальк' },
]

function LayerPanel({
  layers,
  setLayers,
  onShowConfidence,
}: {
  layers: LayerState
  setLayers: (updater: (prev: LayerState) => LayerState) => void
  onShowConfidence: () => void
}) {
  const pct = Math.round(layers.phasesOpacity * 100)
  return (
    <div className={s.layerPanel}>
      <span className="microlabel">Слои</span>

      <div className={s.layerRow}>
        <span className={s.layerName}>Исходное изображение</span>
        <Switch
          on={layers.imageVisible}
          onToggle={() => setLayers((p) => ({ ...p, imageVisible: !p.imageVisible }))}
          label="Исходное изображение"
        />
      </div>

      <div className={s.layerRow}>
        <span className={s.layerName}>Маска фаз</span>
        <Switch
          on={layers.phasesVisible}
          onToggle={() => setLayers((p) => ({ ...p, phasesVisible: !p.phasesVisible }))}
          label="Маска фаз"
        />
      </div>

      <div className={s.opacityBlock}>
        <div className={s.opacityHead}>
          <span className="microlabel">Прозрачность маски</span>
          <span className={s.opacityValue}>{pct}%</span>
        </div>
        <input
          className={s.slider}
          style={{ '--slider-pct': `${pct}%` } as CSSProperties}
          type="range"
          min={0}
          max={100}
          value={pct}
          disabled={!layers.phasesVisible}
          onChange={(e) =>
            setLayers((p) => ({ ...p, phasesOpacity: Number(e.target.value) / 100 }))
          }
          aria-label="Прозрачность маски фаз"
        />
      </div>

      <div className={s.panelDivider} />

      <div className={s.legend}>
        {LEGEND.map((item) => (
          <div key={item.label} className={s.legendItem}>
            <span className={s.legendSwatch} style={{ background: item.color }} />
            {item.label}
          </div>
        ))}
      </div>

      <div className={s.panelDivider} />

      <button className={`btn ${s.confBtn}`} onClick={onShowConfidence} type="button">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
          <path d="M8 3C4.5 3 2 8 2 8s2.5 5 6 5 6-5 6-5-2.5-5-6-5z" strokeLinejoin="round" />
          <circle cx="8" cy="8" r="2" />
        </svg>
        Карта уверенности
      </button>
    </div>
  )
}

/* ================= Правая панель ================= */

function Section({ title, children }: { title?: ReactNode; children: ReactNode }) {
  return (
    <div className={s.section}>
      {title != null && (
        <div className={s.sectionTitle}>
          <span className="microlabel">{title}</span>
        </div>
      )}
      {children}
    </div>
  )
}

function MetricBar({
  label,
  value,
  color,
  scaleMax,
  swatch = true,
}: {
  label: string
  value: number
  color: string
  scaleMax: number
  swatch?: boolean
}) {
  const width = scaleMax > 0 ? Math.min(100, (value / scaleMax) * 100) : 0
  return (
    <div className={s.metricRow}>
      <span className={s.metricLabel}>
        {swatch && <span className={s.legendSwatch} style={{ background: color }} />}
        {label}
      </span>
      <span className={s.metricValue}>{fmtPct(value)}</span>
      <div className={s.metricBarTrack}>
        <div className={s.metricBarFill} style={{ width: `${width}%`, background: color }} />
      </div>
    </div>
  )
}

function TalcBlock({ result }: { result: AnalysisResult }) {
  const { talc_pct, talc_ci } = result.metrics
  const thr = result.params.talc_ore_thr_pct ?? 10
  const [ciLo, ciHi] = talc_ci ?? [talc_pct, talc_pct]
  const domain = Math.max(thr * 2, ciHi * 1.2, talc_pct * 1.2, 12)
  const x = (v: number) => `${Math.min(100, Math.max(0, (v / domain) * 100))}%`
  const above = talc_pct > thr

  return (
    <>
      <div className={s.talcGauge}>
        <div className={s.talcTrack} />
        <div
          className={s.talcCi}
          style={{ left: x(ciLo), width: `calc(${x(ciHi)} - ${x(ciLo)})` }}
          title={`Доверительный интервал: ${fmtNum(ciLo)}–${fmtNum(ciHi)}%`}
        />
        <div className={s.talcThr} style={{ left: x(thr) }}>
          <span className={s.talcThrLabel}>порог {fmtNum(thr)}%</span>
        </div>
        <div className={s.talcValue} style={{ left: x(talc_pct) }} />
      </div>
      <div className={s.talcCaption}>
        <span>
          тальк <b style={{ color: 'var(--text-0)' }}>{fmtPct(talc_pct)}</b>
        </span>
        <span>
          ДИ 95%: {fmtNum(ciLo)}–{fmtNum(ciHi)}%
        </span>
      </div>
      <div
        className={s.talcVerdict}
        style={{ color: above ? 'var(--phase-talc)' : 'var(--ok)' }}
      >
        {above
          ? `Выше порога ${fmtNum(thr)}% — руда оталькованная`
          : `Ниже порога ${fmtNum(thr)}% — оталькованность не подтверждена`}
      </div>
    </>
  )
}

function SidePanel({
  detail,
  result,
}: {
  detail: AnalysisDetail
  result: AnalysisResult
}) {
  const m = result.metrics
  const heroColor = oreClassColor(result.ore_class)
  const phaseScale =
    Math.max(m.sulfide_total_pct, m.ordinary_pct, m.fine_pct, m.talc_pct, m.gray_phase_pct, 1) *
    1.1

  return (
    <aside className={s.sidePanel}>
      <div className={`${s.section} ${s.hero}`} style={{ '--hero-color': heroColor } as CSSProperties}>
        <span className="microlabel">Класс руды</span>
        <div className={s.heroClass}>{result.ore_class}</div>
        <div className={s.heroMeta}>
          <span>
            уверенность <b>{fmtNum(result.confidence * 100, 0)}%</b>
          </span>
          <span>
            модель <b>{result.params.model_version}</b>
          </span>
        </div>
      </div>

      <Section title="Заключение">
        <p className={s.conclusion}>{result.conclusion}</p>
      </Section>

      <Section title="Фазовый состав">
        <MetricBar label="Сульфиды, всего" value={m.sulfide_total_pct} color="var(--accent)" scaleMax={phaseScale} />
        <MetricBar label="Обычные срастания" value={m.ordinary_pct} color="var(--phase-ordinary)" scaleMax={phaseScale} />
        <MetricBar label="Тонкие срастания" value={m.fine_pct} color="var(--phase-fine)" scaleMax={phaseScale} />
        <MetricBar label="Тальк" value={m.talc_pct} color="var(--phase-talc)" scaleMax={phaseScale} />
        <MetricBar label="Серая фаза" value={m.gray_phase_pct} color="var(--phase-gray)" scaleMax={phaseScale} />

        <div>
          <div className="microlabel" style={{ marginBottom: 8 }}>
            Структура сульфидов
          </div>
          <div className={s.splitBar}>
            <div
              className={s.splitSeg}
              style={{ width: `${m.ordinary_share}%`, background: 'var(--phase-ordinary)' }}
              title={`Обычные: ${fmtPct(m.ordinary_share)}`}
            />
            <div
              className={s.splitSeg}
              style={{ width: `${m.fine_share}%`, background: 'var(--phase-fine)' }}
              title={`Тонкие: ${fmtPct(m.fine_share)}`}
            />
          </div>
          <div className={s.splitCaption} style={{ marginTop: 6 }}>
            <span>обычные {fmtPct(m.ordinary_share)}</span>
            <span>тонкие {fmtPct(m.fine_share)}</span>
          </div>
        </div>
      </Section>

      <Section title="Оталькованность">
        <TalcBlock result={result} />
      </Section>

      <Section title="Гранулометрия сульфидов · ECD">
        <EcdChart granulometry={result.granulometry} />
        <div className={s.statTiles}>
          <div className={s.statTile}>
            <span className={s.statTileLabel}>P50 · медиана</span>
            <span className={s.statTileValue}>
              {fmtNum(result.granulometry.ecd_p50)} <small>мкм</small>
            </span>
          </div>
          <div className={s.statTile}>
            <span className={s.statTileLabel}>P80</span>
            <span className={s.statTileValue}>
              {fmtNum(result.granulometry.ecd_p80)} <small>мкм</small>
            </span>
          </div>
        </div>
      </Section>

      <Section title="Объяснение решения">
        <div>
          {result.explanation.map((item, i) => (
            <div key={i} className={s.factor}>
              <span className={s.factorText}>{item.factor}</span>
              <span
                className={s.weightChip}
                style={{ '--chip-color': weightColor(item.weight) } as CSSProperties}
              >
                {item.weight}
              </span>
            </div>
          ))}
          {result.explanation.length === 0 && (
            <span style={{ color: 'var(--text-2)', fontSize: 12 }}>Факторы не переданы</span>
          )}
        </div>
      </Section>

      {Object.keys(result.features).length > 0 && (
        <Section title="Морфометрические признаки">
          <div className={s.featureGrid}>
            {Object.entries(result.features).map(([key, value]) => (
              <div key={key} className={s.statTile}>
                <span className={s.statTileLabel}>{FEATURE_LABELS[key] ?? key}</span>
                <span className={s.statTileValue} style={{ fontSize: 14 }}>
                  {fmtNum(value, 2)}
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      <Section title="Экспорт">
        <div className={s.exportRow}>
          <a className="btn" href={urls.reportPdf(detail.id)} download>
            PDF
          </a>
          <a className="btn" href={urls.metricsCsv(detail.id)} download>
            CSV
          </a>
          <a className="btn" href={urls.maskGeojson(detail.id)} download>
            GeoJSON
          </a>
        </div>
      </Section>

      <div className={s.panelFooter}>
        анализ № {detail.id}
        <br />
        {detail.image && (
          <>
            {fmtNum(detail.image.width, 0)} × {fmtNum(detail.image.height, 0)} px · рабочий
            масштаб {fmtNum(detail.image.work_scale * 100, 0)}%
            <br />
          </>
        )}
        порог оталькованности {fmtNum(result.params.talc_ore_thr_pct ?? 10)}%
      </div>
    </aside>
  )
}

/* ================= Полноэкранные состояния ================= */

function ProcessingScreen({
  detail,
  stage,
  percent,
}: {
  detail: AnalysisDetail
  stage: string
  percent: number
}) {
  return (
    <div className={s.stateScreen}>
      <svg className={s.processingReticle} width="72" height="72" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.3" aria-hidden>
        <circle cx="24" cy="24" r="16" strokeDasharray="6 5" />
        <circle cx="24" cy="24" r="8" opacity="0.6" />
        <path d="M24 2v6M24 40v6M2 24h6M40 24h6" strokeLinecap="round" />
      </svg>
      <div className={s.processingPercent}>{Math.round(percent)}%</div>
      <div className={s.processingStage}>{stage}</div>
      <div className={s.processingTrack}>
        <ProgressBar percent={percent} indeterminate={percent === 0} />
      </div>
      <div className={s.processingFile}>
        {detail.file_name} · поставлен {fmtDateTime(detail.created_at)}
      </div>
    </div>
  )
}

/* ================= Страница ================= */

export function AnalysisPage({
  id,
  onBack,
  onAnnotate,
}: {
  id: string
  onBack: () => void
  onAnnotate?: (id: string) => void
}) {
  const [detail, setDetail] = useState<AnalysisDetail | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [layers, setLayersState] = useState<LayerState>({
    imageVisible: true,
    phasesVisible: true,
    phasesOpacity: 0.6,
  })
  const [confOpen, setConfOpen] = useState(false)
  const [confLoaded, setConfLoaded] = useState(false)
  const [flagged, setFlagged] = useState(false)
  const [flagOpen, setFlagOpen] = useState(false)
  const [reason, setReason] = useState('wrong_class')
  const [note, setNote] = useState('')
  const [pendingDelete, setPendingDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .getFeedback(id)
      .then((fb) => !cancelled && setFlagged(fb.flagged))
      .catch(() => undefined) // при сбое не трогаем флаг (не форсим false)
    return () => {
      cancelled = true
    }
  }, [id])

  const submitFlag = async () => {
    setActionError(null)
    try {
      await api.flag(id, reason, note)
      setFlagged(true)
      setFlagOpen(false)
      setNote('')
    } catch (e) {
      setActionError(errorMessage(e))
    }
  }
  const removeFlag = async () => {
    setActionError(null)
    try {
      await api.unflag(id)
      setFlagged(false)
    } catch (e) {
      setActionError(errorMessage(e))
    }
  }

  const load = useCallback(async () => {
    try {
      setDetail(await api.getAnalysis(id))
      setFetchError(null)
    } catch (e) {
      setFetchError(errorMessage(e))
    }
  }, [id])

  useEffect(() => {
    void load()
  }, [load])

  const active = detail?.status === 'queued' || detail?.status === 'running'
  const progress = useProgress(active ? id : null, () => void load())

  // Резервный опрос на случай проблем с SSE.
  useEffect(() => {
    if (!active) return
    const t = window.setInterval(() => void load(), 3000)
    return () => window.clearInterval(t)
  }, [active, load])

  const del = () => {
    setActionError(null)
    setPendingDelete(true)
  }
  const confirmDelete = async () => {
    setDeleting(true)
    setActionError(null)
    try {
      await api.deleteAnalysis(id)
      onBack()
    } catch (e) {
      setActionError(errorMessage(e))
      setDeleting(false)
      setPendingDelete(false)
    }
  }

  const setLayers = useCallback(
    (updater: (prev: LayerState) => LayerState) => setLayersState(updater),
    [],
  )

  return (
    <div className={s.page}>
      <div className={s.toolbar}>
        <button className={s.backBtn} onClick={onBack} type="button">
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
            <path d="M9 2L4 7l5 5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          К журналу
        </button>

        <div className={s.fileMeta}>
          <span className={s.fileName}>{detail?.file_name ?? '…'}</span>
          {detail && (
            <>
              <span className={s.metaChip}>№ {detail.id}</span>
              <span className={s.metaChip}>{fmtDateTime(detail.created_at)}</span>
              {detail.elapsed_s != null && (
                <span className={s.metaChip}>⏱ {fmtElapsed(detail.elapsed_s)}</span>
              )}
              <span
                className={s.metaChip}
                style={{ color: STATUS_COLOR[detail.status], display: 'inline-flex', alignItems: 'center', gap: 6 }}
              >
                <Led color={STATUS_COLOR[detail.status]} pulse={active} />
                {STATUS_LABEL[detail.status]}
              </span>
            </>
          )}
        </div>

        {flagged ? (
          <>
            <span className={s.flagChip} title="Снимок в очереди на доработку">
              <Led color="var(--accent)" /> На доработке
            </span>
            {onAnnotate && (
              <button className="btn btn-primary" onClick={() => onAnnotate(id)} type="button">
                Разметить
              </button>
            )}
            <button className="btn btn-ghost" onClick={() => void removeFlag()} type="button">
              Снять флаг
            </button>
          </>
        ) : (
          <button className="btn" onClick={() => setFlagOpen((v) => !v)} type="button"
                  title="Отправить снимок на доработку и разметку для дообучения">
            На доработку
          </button>
        )}

        <button className="btn btn-ghost btn-danger" onClick={() => void del()} type="button">
          Удалить
        </button>
      </div>

      {actionError && <div className={s.actionError}>{actionError}</div>}

      {flagOpen && !flagged && (
        <div className={s.flagForm}>
          <span className={s.flagFormLabel} id={`flag-reason-${id}`}>Причина:</span>
          <select
            className={s.flagSelect}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            aria-labelledby={`flag-reason-${id}`}
          >
            <option value="wrong_class">Неверный класс руды</option>
            <option value="bad_talc">Ошибка по тальку</option>
            <option value="bad_segmentation">Плохая сегментация фаз</option>
            <option value="poor_quality">Низкое качество снимка</option>
            <option value="other">Другое</option>
          </select>
          <input
            className={s.flagNote}
            type="text"
            placeholder="Комментарий (необязательно)"
            aria-label="Комментарий к доработке"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <button className="btn btn-primary" onClick={() => void submitFlag()} type="button">
            Отправить на доработку
          </button>
          <button className="btn btn-ghost" onClick={() => setFlagOpen(false)} type="button">
            Отмена
          </button>
        </div>
      )}

      {/* --- Состояния --- */}
      {!detail && !fetchError && (
        <div className={s.stateScreen}>
          <Spinner size={30} />
          <span style={{ color: 'var(--text-2)', font: '400 12px/1.4 var(--font-mono)' }}>
            загрузка анализа…
          </span>
        </div>
      )}

      {!detail && fetchError && (
        <div className={s.stateScreen}>
          <div className={s.errorBox}>Не удалось загрузить анализ: {fetchError}</div>
          <button className="btn" onClick={() => void load()} type="button">
            Повторить
          </button>
        </div>
      )}

      {detail && active && (
        <ProcessingScreen
          detail={detail}
          stage={progress?.stage ?? (detail.status === 'queued' ? 'ожидание очереди' : 'обработка')}
          percent={progress?.percent ?? 0}
        />
      )}

      {detail && detail.status === 'error' && (
        <div className={s.stateScreen}>
          <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="var(--err)" strokeWidth="1.2" aria-hidden>
            <path d="M12 3L2.5 20h19L12 3z" strokeLinejoin="round" />
            <path d="M12 9.5v4.5" strokeLinecap="round" />
            <circle cx="12" cy="17" r="0.9" fill="var(--err)" stroke="none" />
          </svg>
          <div style={{ font: '500 15px/1.3 var(--font-display)', color: 'var(--err)', letterSpacing: '0.04em' }}>
            Анализ завершился с ошибкой
          </div>
          <div className={s.errorBox}>{detail.error ?? 'Причина не указана сервером.'}</div>
          <button className="btn" onClick={onBack} type="button">
            Вернуться к журналу
          </button>
        </div>
      )}

      {detail && detail.status === 'done' && !detail.result && (
        <div className={s.stateScreen}>
          <div className={s.errorBox}>Анализ завершён, но результат отсутствует в ответе сервера.</div>
        </div>
      )}

      {detail && detail.status === 'done' && detail.result && (
        <div className={s.body}>
          <div className={s.viewerArea}>
            <OsdViewer analysisId={detail.id} layers={layers} />
            <LayerPanel
              layers={layers}
              setLayers={setLayers}
              onShowConfidence={() => {
                setConfLoaded(false)
                setConfOpen(true)
              }}
            />
          </div>
          <SidePanel detail={detail} result={detail.result} />
        </div>
      )}

      {confOpen && detail && (
        <Modal title="Карта уверенности модели" width={860} onClose={() => setConfOpen(false)}>
          <div className={s.confImgWrap}>
            {!confLoaded && <Spinner size={26} />}
            <img
              className={s.confImg}
              style={confLoaded ? undefined : { display: 'none' }}
              src={urls.confidence(detail.id)}
              alt="Карта уверенности сегментации"
              onLoad={() => setConfLoaded(true)}
              onError={() => setConfLoaded(true)}
            />
            <div className={s.confHint}>
              Тёплые области — низкая уверенность сегментации талька; рабочий масштаб ≤ 2048 px.
            </div>
          </div>
        </Modal>
      )}

      {pendingDelete && (
        <Modal
          title="Удалить анализ?"
          width={420}
          onClose={deleting ? () => undefined : () => setPendingDelete(false)}
        >
          <p style={{ font: '400 13.5px/1.6 var(--font-ui)', color: 'var(--text-1)', margin: 0 }}>
            Удалить анализ вместе с результатами? Действие необратимо — снимок, маски и
            отчёты будут удалены.
          </p>
          {actionError && <div className={s.actionError} style={{ margin: '12px 0 0' }}>{actionError}</div>}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 22 }}>
            <button className="btn btn-ghost" onClick={() => setPendingDelete(false)} disabled={deleting} type="button">
              Отмена
            </button>
            <button className="btn btn-danger" onClick={() => void confirmDelete()} disabled={deleting} type="button">
              {deleting ? (
                <>
                  <Spinner size={13} /> удаление…
                </>
              ) : (
                'Удалить'
              )}
            </button>
          </div>
        </Modal>
      )}
    </div>
  )
}
