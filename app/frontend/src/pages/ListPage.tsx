import { useCallback, useEffect, useMemo, useState, type CSSProperties } from 'react'
import { api, errorMessage, urls } from '../api/client'
import type { AnalysisStatus, AnalysisSummary } from '../api/types'
import { useProgress } from '../hooks/useProgress'
import {
  fmtDateTime,
  fmtElapsed,
  fmtPct,
  oreClassColor,
  STATUS_COLOR,
  STATUS_LABEL,
} from '../lib/format'
import { Led, Modal, ProgressBar, Skeleton, Spinner } from '../components/ui'
import { NewAnalysisModal } from '../components/NewAnalysisModal'
import type { UploadJob, UploadRequest } from '../hooks/useUploads'
import s from './ListPage.module.css'

const ACTIVE: AnalysisStatus[] = ['queued', 'running']
/** Цвет прогресса ЗАГРУЗКИ на сервер (обработка идёт цветом статуса). */
const UPLOAD_COLOR = '#2f6fd6'

/* ---------- Строка фоновой загрузки (файл ещё летит на сервер) ---------- */

function UploadRow({ job, onDismiss }: { job: UploadJob; onDismiss: (tempId: string) => void }) {
  const err = job.phase === 'error'
  return (
    <div className={s.row}>
      <span className={s.checkCell} />
      <div className={s.thumbWrap}>
        {err ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--err)" strokeWidth="1.5" aria-hidden>
            <circle cx="12" cy="12" r="9" />
            <path d="M12 7.5v5M12 16h.01" strokeLinecap="round" />
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
            <path d="M12 16V5m0 0L7.5 9.5M12 5l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M4 17v2a1 1 0 001 1h14a1 1 0 001-1v-2" strokeLinecap="round" />
          </svg>
        )}
      </div>
      <div className={s.fileCol}>
        <span className={s.fileName}>{job.name}</span>
        <span className={s.fileId}>{err ? 'загрузка не удалась' : 'загрузка на сервер'}</span>
      </div>
      <div className={s.statusCell}>
        {err ? (
          <span className={s.statusLine} style={{ color: 'var(--err)' }}>
            <Led color="var(--err)" /> ошибка загрузки
          </span>
        ) : (
          <>
            <span className={s.statusLine} style={{ color: UPLOAD_COLOR }}>
              <Led color={UPLOAD_COLOR} pulse /> загрузка
              {job.isFile && (
                <span className="num" style={{ color: 'var(--text-1)' }}>{job.pct}%</span>
              )}
            </span>
            <ProgressBar percent={job.pct} indeterminate={!job.isFile || job.pct === 0} color={UPLOAD_COLOR} />
            <span className={s.stage}>{job.isFile ? 'передача файла на сервер' : 'постановка в очередь'}</span>
          </>
        )}
      </div>
      <div><span className={s.numDim}>—</span></div>
      <div className={`${s.numCell} ${s.numDim}`}>—</div>
      <div className={`${s.numCell} ${s.numDim} ${s.hideNarrow}`}>—</div>
      <div className={`${s.numCell} ${s.numDim} ${s.hideNarrow}`}>—</div>
      <div className={s.dateCell}>{err ? job.error?.slice(0, 40) : 'сейчас'}</div>
      {err ? (
        <button className={s.deleteBtn} onClick={() => onDismiss(job.tempId)} title="Убрать" type="button">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
            <path d="M2 2l8 8M10 2l-8 8" strokeLinecap="round" />
          </svg>
        </button>
      ) : (
        <span />
      )}
    </div>
  )
}

function pluralAnalyses(n: number): string {
  const d = n % 10
  const h = n % 100
  if (d === 1 && h !== 11) return 'анализ'
  if (d >= 2 && d <= 4 && (h < 12 || h > 14)) return 'анализа'
  return 'анализов'
}

/* ---------- Живой прогресс строки (SSE) ---------- */

function LiveStatus({ item, onFinal }: { item: AnalysisSummary; onFinal: () => void }) {
  const progress = useProgress(item.id, onFinal)
  const pct = progress?.percent ?? 0
  const stage =
    progress?.stage ?? (item.status === 'queued' ? 'ожидание очереди' : 'подготовка…')

  return (
    <div className={s.statusCell}>
      <span className={s.statusLine} style={{ color: STATUS_COLOR[item.status] }}>
        <Led color={STATUS_COLOR[item.status]} pulse />
        {STATUS_LABEL[item.status]}
        <span className="num" style={{ color: 'var(--text-1)' }}>
          {Math.round(pct)}%
        </span>
      </span>
      <ProgressBar percent={pct} indeterminate={item.status === 'queued' && pct === 0} />
      <span className={s.stage}>{stage}</span>
    </div>
  )
}

function StaticStatus({ status }: { status: AnalysisStatus }) {
  return (
    <div className={s.statusCell}>
      <span className={s.statusLine} style={{ color: STATUS_COLOR[status] }}>
        <Led color={STATUS_COLOR[status]} />
        {STATUS_LABEL[status]}
      </span>
    </div>
  )
}

/* ---------- Строка журнала ---------- */

function Row({
  item,
  onOpen,
  onChanged,
  selected,
  onToggle,
  onDelete,
}: {
  item: AnalysisSummary
  onOpen: (id: string) => void
  onChanged: () => void
  selected: boolean
  onToggle: (id: string) => void
  onDelete: (id: string) => void
}) {
  const [thumbFailed, setThumbFailed] = useState(false)
  // Живой SSE-поток держим ТОЛЬКО для выполняющегося анализа: очередь
  // обновляется общим 4-секундным опросом, иначе пачка задач упирается в
  // лимит одновременных соединений браузера (~6 на хост по HTTP/1.1).
  const live = item.status === 'running'

  return (
    <div
      className={`${s.row} ${selected ? s.rowSelected : ''}`}
      onClick={() => onOpen(item.id)}
    >
      <label className={s.checkCell} onClick={(e) => e.stopPropagation()}>
        <input
          type="checkbox"
          className={s.checkbox}
          checked={selected}
          onChange={() => onToggle(item.id)}
          aria-label={`Выбрать «${item.file_name}»`}
        />
      </label>

      <div className={s.thumbWrap}>
        {thumbFailed || item.status === 'queued' ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
            <circle cx="12" cy="12" r="7.5" />
            <path d="M12 1.5v4M12 18.5v4M1.5 12h4M18.5 12h4" strokeLinecap="round" />
          </svg>
        ) : (
          <img
            className={s.thumb}
            src={urls.preview(item.id)}
            alt=""
            loading="lazy"
            onError={() => setThumbFailed(true)}
          />
        )}
      </div>

      <button
        className={s.openBtn}
        onClick={(e) => {
          e.stopPropagation()
          onOpen(item.id)
        }}
        type="button"
      >
        <span className={s.fileName}>{item.file_name}</span>
        <span className={s.fileId}>№ {item.id}</span>
      </button>

      {live ? (
        <LiveStatus item={item} onFinal={onChanged} />
      ) : (
        <StaticStatus status={item.status} />
      )}

      <div>
        {item.ore_class ? (
          <span
            className={s.classBadge}
            style={{ '--badge-color': oreClassColor(item.ore_class) } as CSSProperties}
          >
            {item.ore_class}
          </span>
        ) : (
          <span className={s.numDim}>—</span>
        )}
      </div>

      <div className={`${s.numCell} ${item.talc_pct == null ? s.numDim : ''}`}>
        {fmtPct(item.talc_pct)}
      </div>
      <div className={`${s.numCell} ${item.sulfide_total_pct == null ? s.numDim : ''} ${s.hideNarrow}`}>
        {fmtPct(item.sulfide_total_pct)}
      </div>
      <div className={`${s.numCell} ${s.numDim} ${s.hideNarrow}`}>{fmtElapsed(item.elapsed_s)}</div>
      <div className={s.dateCell}>{fmtDateTime(item.created_at)}</div>

      <button
        className={s.deleteBtn}
        onClick={(e) => {
          e.stopPropagation()
          onDelete(item.id)
        }}
        title="Удалить анализ"
        type="button"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
          <path d="M2.5 4h11M6.5 4V2.5h3V4M4 4l.8 9.5h6.4L12 4M6.5 7v4M9.5 7v4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </div>
  )
}

/* ---------- Страница ---------- */

export function ListPage({
  onOpen,
  uploads,
  startUploads,
  dismissUpload,
  uploadsDoneTick,
}: {
  onOpen: (id: string) => void
  uploads: UploadJob[]
  startUploads: (items: UploadRequest[], threshold: number) => void
  dismissUpload: (tempId: string) => void
  uploadsDoneTick: number
}) {
  const [items, setItems] = useState<AnalysisSummary[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [pendingDelete, setPendingDelete] = useState<string[] | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const list = await api.listAnalyses()
      setItems(list)
      setLoadError(null)
      // выкидываем из выделения исчезнувшие анализы
      setSelected((prev) => {
        if (prev.size === 0) return prev
        const live = new Set(list.map((a) => a.id))
        const next = new Set([...prev].filter((id) => live.has(id)))
        return next.size === prev.size ? prev : next
      })
    } catch (e) {
      setLoadError(errorMessage(e))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  // Файл догрузился → анализ уже в очереди на бэкенде, обновляем список.
  useEffect(() => {
    if (uploadsDoneTick > 0) void load()
  }, [uploadsDoneTick, load])

  const hasActive = useMemo(
    () => items?.some((a) => ACTIVE.includes(a.status)) ?? false,
    [items],
  )

  useEffect(() => {
    if (!hasActive) return
    const t = window.setInterval(() => void load(), 4000)
    return () => window.clearInterval(t)
  }, [hasActive, load])

  const doneCount = items?.filter((a) => a.status === 'done').length ?? 0

  const sorted = useMemo(
    () =>
      items
        ? [...items].sort(
            (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
          )
        : null,
    [items],
  )

  const ids = useMemo(() => sorted?.map((a) => a.id) ?? [], [sorted])
  const allSelected = ids.length > 0 && ids.every((id) => selected.has(id))
  const someSelected = selected.size > 0 && !allSelected

  const toggle = (id: string) =>
    setSelected((prev) => {
      const n = new Set(prev)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })
  const toggleAll = () => setSelected(allSelected ? new Set() : new Set(ids))
  const clearSel = () => setSelected(new Set())

  const confirmDelete = async () => {
    if (!pendingDelete) return
    setDeleting(true)
    setDeleteError(null)
    let fail = 0
    for (const id of pendingDelete) {
      try {
        await api.deleteAnalysis(id)
      } catch {
        fail += 1
      }
    }
    setDeleting(false)
    setPendingDelete(null)
    setSelected(new Set())
    if (fail > 0) setDeleteError(`Не удалось удалить ${fail} из ${pendingDelete.length}`)
    await load()
  }

  const pendingCount = pendingDelete?.length ?? 0
  const pendingSingle =
    pendingCount === 1 ? items?.find((i) => i.id === pendingDelete![0]) ?? null : null

  return (
    <main className={s.page}>
      <div className={s.pageHead}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>Журнал анализов</h1>
          <div className={s.subtitle}>
            {items ? (
              <>
                всего <b>{items.length}</b> · завершено <b>{doneCount}</b>
                {hasActive && (
                  <>
                    {' '}
                    · в работе <b>{items.filter((a) => ACTIVE.includes(a.status)).length}</b>
                  </>
                )}
                {uploads.length > 0 && (
                  <>
                    {' '}
                    · загружается <b>{uploads.length}</b>
                  </>
                )}
              </>
            ) : (
              'загрузка…'
            )}
          </div>
        </div>
        <div className={s.actions}>
          {doneCount > 0 && (
            <a className="btn" href={urls.batchCsv} download>
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
                <path d="M8 2v8m0 0L5 7m3 3l3-3M2.5 12.5v1a1 1 0 001 1h9a1 1 0 001-1v-1" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Сводный CSV
            </a>
          )}
          <button className="btn btn-primary" onClick={() => setModalOpen(true)} type="button">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
              <path d="M8 2.5v11M2.5 8h11" strokeLinecap="round" />
            </svg>
            Новый анализ
          </button>
        </div>
      </div>

      {loadError && items && (
        <div className={s.offlineBanner}>
          <Led color="var(--err)" pulse />
          Нет связи с сервером — данные могли устареть. {loadError}
        </div>
      )}
      {deleteError && (
        <div className={s.offlineBanner}>
          <Led color="var(--err)" />
          {deleteError}
        </div>
      )}

      {selected.size > 0 && (
        <div className={s.selectBar}>
          <span className={s.selectCount}>
            Выбрано <b>{selected.size}</b> {pluralAnalyses(selected.size)}
          </span>
          <div className={s.selectActions}>
            <button className="btn btn-ghost" onClick={clearSel} type="button">
              Снять выделение
            </button>
            <button
              className={`btn ${s.dangerBtn}`}
              onClick={() => setPendingDelete([...selected])}
              type="button"
            >
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
                <path d="M2.5 4h11M6.5 4V2.5h3V4M4 4l.8 9.5h6.4L12 4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Удалить выбранные
            </button>
          </div>
        </div>
      )}

      <div className={s.log}>
        {sorted === null && loadError === null && (
          <>
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className={s.skeletonRow}>
                <span />
                <Skeleton width={74} height={48} />
                <Skeleton height={14} style={{ maxWidth: 220 }} />
                <Skeleton height={12} style={{ maxWidth: 90 }} />
                <Skeleton height={22} style={{ maxWidth: 130, borderRadius: 999 }} />
                <Skeleton height={12} />
                <Skeleton height={12} />
                <Skeleton height={12} />
                <Skeleton height={12} />
                <span />
              </div>
            ))}
          </>
        )}

        {sorted === null && loadError !== null && (
          <div className={s.dead}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--err)" strokeWidth="1.2" aria-hidden>
              <circle cx="12" cy="12" r="9" />
              <path d="M8.5 9.5l7 7M15.5 9.5l-7 7" strokeLinecap="round" opacity="0.7" />
            </svg>
            <div className={s.deadTitle}>Сервер недоступен</div>
            <div className={s.deadText}>
              Не удалось получить список анализов. Убедитесь, что бэкенд запущен, и
              повторите попытку.
            </div>
            <button className="btn" onClick={() => void load()} type="button">
              Повторить
            </button>
          </div>
        )}

        {sorted !== null && sorted.length === 0 && uploads.length === 0 && (
          <div className={s.empty}>
            <svg className={s.emptyIcon} width="56" height="56" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.2" aria-hidden>
              <circle cx="24" cy="24" r="15" strokeDasharray="4 4" />
              <circle cx="24" cy="24" r="8" />
              <path d="M24 2v7M24 39v7M2 24h7M39 24h7" strokeLinecap="round" />
              <circle cx="24" cy="24" r="1.8" fill="currentColor" stroke="none" />
            </svg>
            <div className={s.emptyTitle}>Ни одного анализа</div>
            <div className={s.emptyText}>
              Загрузите панорамное изображение полированного шлифа — система выполнит
              сегментацию фаз, оценит оталькованность и определит класс руды.
            </div>
            <button className="btn btn-primary" onClick={() => setModalOpen(true)} type="button">
              Загрузить первый снимок
            </button>
          </div>
        )}

        {sorted !== null && (sorted.length > 0 || uploads.length > 0) && (
          <>
            <div className={s.logHead}>
              <label className={s.checkCell}>
                <input
                  type="checkbox"
                  className={s.checkbox}
                  checked={allSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = someSelected
                  }}
                  onChange={toggleAll}
                  aria-label="Выбрать все"
                />
              </label>
              <span className="microlabel" />
              <span className="microlabel">Файл</span>
              <span className="microlabel">Статус</span>
              <span className="microlabel">Класс руды</span>
              <span className="microlabel" style={{ textAlign: 'right' }}>
                Тальк
              </span>
              <span className={`microlabel ${s.hideNarrow}`} style={{ textAlign: 'right' }}>
                Сульфиды
              </span>
              <span className={`microlabel ${s.hideNarrow}`} style={{ textAlign: 'right' }}>
                Время
              </span>
              <span className="microlabel">Создан</span>
              <span className="microlabel" />
            </div>
            {uploads.map((u) => (
              <UploadRow key={u.tempId} job={u} onDismiss={dismissUpload} />
            ))}
            {sorted.map((item) => (
              <Row
                key={item.id}
                item={item}
                onOpen={onOpen}
                onChanged={() => void load()}
                selected={selected.has(item.id)}
                onToggle={toggle}
                onDelete={(id) => setPendingDelete([id])}
              />
            ))}
          </>
        )}
      </div>

      {modalOpen && (
        <NewAnalysisModal
          onClose={() => setModalOpen(false)}
          onSubmit={(reqs, threshold) => {
            startUploads(reqs, threshold)
            setModalOpen(false)
          }}
        />
      )}

      {pendingDelete && (
        <Modal
          title={pendingCount === 1 ? 'Удалить анализ?' : 'Удалить анализы?'}
          width={440}
          onClose={deleting ? () => undefined : () => setPendingDelete(null)}
        >
          <p className={s.confirmText}>
            {pendingSingle ? (
              <>
                Удалить анализ <b>«{pendingSingle.file_name}»</b>?
              </>
            ) : (
              <>
                Удалить <b>{pendingCount}</b> {pluralAnalyses(pendingCount)}?
              </>
            )}
            <br />
            Действие необратимо — снимок, маски и отчёты будут удалены.
          </p>
          <div className={s.modalFooter}>
            <button
              className="btn btn-ghost"
              onClick={() => setPendingDelete(null)}
              disabled={deleting}
              type="button"
            >
              Отмена
            </button>
            <button
              className={`btn ${s.dangerBtn}`}
              onClick={() => void confirmDelete()}
              disabled={deleting}
              type="button"
            >
              {deleting ? (
                <>
                  <Spinner size={13} /> удаление…
                </>
              ) : pendingCount === 1 ? (
                'Удалить'
              ) : (
                `Удалить ${pendingCount}`
              )}
            </button>
          </div>
        </Modal>
      )}
    </main>
  )
}
