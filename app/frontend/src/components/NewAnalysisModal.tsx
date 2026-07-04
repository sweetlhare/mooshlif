import { useEffect, useRef, useState, type DragEvent } from 'react'
import { api, errorMessage } from '../api/client'
import type { DemoImage } from '../api/types'
import { fmtNum } from '../lib/format'
import { Modal, ProgressBar, Skeleton, Spinner } from './ui'
import s from './NewAnalysisModal.module.css'

const ACCEPT_EXT = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
const DEFAULT_THRESHOLD = '10'

function isAcceptedFile(f: File): boolean {
  const name = f.name.toLowerCase()
  return ACCEPT_EXT.some((ext) => name.endsWith(ext))
}

function fmtBytes(n: number): string {
  if (n < 1024 * 1024) return `${fmtNum(n / 1024, 0)} КБ`
  if (n < 1024 * 1024 * 1024) return `${fmtNum(n / (1024 * 1024), 1)} МБ`
  return `${fmtNum(n / (1024 * 1024 * 1024), 2)} ГБ`
}

export function NewAnalysisModal({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (id: string) => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [demo, setDemo] = useState<DemoImage | null>(null)
  const [demos, setDemos] = useState<DemoImage[] | null>(null)
  const [demosError, setDemosError] = useState(false)
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  const [dragOver, setDragOver] = useState(false)
  const [busy, setBusy] = useState(false)
  const [uploadPct, setUploadPct] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let cancelled = false
    api
      .demoImages()
      .then((list) => !cancelled && setDemos(list))
      .catch(() => !cancelled && setDemosError(true))
    return () => {
      cancelled = true
    }
  }, [])

  const pickFile = (f: File) => {
    if (!isAcceptedFile(f)) {
      setError(`Неподдерживаемый формат. Ожидается: ${ACCEPT_EXT.join(', ')}`)
      return
    }
    setError(null)
    setFile(f)
    setDemo(null)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) pickFile(f)
  }

  const parsedThreshold = (() => {
    const v = Number.parseFloat(threshold.replace(',', '.'))
    return Number.isFinite(v) && v > 0 && v < 100 ? v : null
  })()

  const canSubmit = (file !== null || demo !== null) && parsedThreshold !== null && !busy

  const submit = async () => {
    if (!canSubmit || parsedThreshold === null) return
    setBusy(true)
    setError(null)
    try {
      const res = await api.createAnalysis({
        file: file ?? undefined,
        serverPath: demo?.server_path,
        talcThresholdPct: parsedThreshold,
        onUploadProgress: file ? (f) => setUploadPct(f * 100) : undefined,
      })
      onCreated(res.id)
    } catch (e) {
      setError(errorMessage(e))
      setBusy(false)
      setUploadPct(null)
    }
  }

  return (
    <Modal title="Новый анализ" onClose={busy ? () => undefined : onClose} width={580}>
      {/* --- Загрузка файла --- */}
      <div
        className={[
          s.dropzone,
          dragOver ? s.dropzoneActive : '',
          file ? s.dropzoneSelected : '',
        ].join(' ')}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_EXT.join(',')}
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) pickFile(f)
            e.target.value = ''
          }}
        />
        {file ? (
          <div className={s.selectedFile}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
              <path d="M13.5 4.5l-7 7L2.5 8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className={s.selectedName}>{file.name}</span>
            <span style={{ color: 'var(--text-2)' }}>{fmtBytes(file.size)}</span>
            <button
              className={s.clearFile}
              onClick={(e) => {
                e.stopPropagation()
                setFile(null)
              }}
              title="Убрать файл"
              type="button"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
                <path d="M2 2l8 8M10 2l-8 8" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        ) : (
          <>
            <svg className={s.dropIcon} width="34" height="34" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
              <path d="M16 21V7m0 0l-5.5 5.5M16 7l5.5 5.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M5 21.5v3a2 2 0 002 2h18a2 2 0 002-2v-3" strokeLinecap="round" />
            </svg>
            <div className={s.dropTitle}>
              Перетащите панораму шлифа или <em>выберите файл</em>
            </div>
            <div className={s.dropHint}>TIFF, PNG, JPEG, BMP — вплоть до гигапикселя</div>
          </>
        )}
      </div>

      {/* --- Демо-набор --- */}
      <div className={s.divider}>
        <span className="microlabel">или демо-набор</span>
      </div>

      {demosError ? (
        <div className={s.dropHint} style={{ textAlign: 'center', padding: '6px 0' }}>
          Демо-набор недоступен
        </div>
      ) : demos === null ? (
        <div className={s.demoList}>
          <Skeleton height={42} />
          <Skeleton height={42} />
        </div>
      ) : demos.length === 0 ? (
        <div className={s.dropHint} style={{ textAlign: 'center', padding: '6px 0' }}>
          Демо-набор пуст
        </div>
      ) : (
        <div className={s.demoList}>
          {demos.map((d) => (
            <button
              key={d.server_path}
              className={`${s.demoItem} ${demo?.server_path === d.server_path ? s.demoItemActive : ''}`}
              onClick={() => {
                setDemo(demo?.server_path === d.server_path ? null : d)
                setFile(null)
                setError(null)
              }}
              type="button"
            >
              <span className={s.demoName}>{d.name}</span>
              <span className={s.demoSize}>
                {fmtNum(d.size[0], 0)} × {fmtNum(d.size[1], 0)} px
              </span>
            </button>
          ))}
        </div>
      )}

      {/* --- Параметры --- */}
      <div className={s.paramRow}>
        <div className={s.paramLabel}>
          <span className={s.paramTitle}>Порог оталькованности</span>
          <span className={s.paramHint}>
            Доля талька, выше которой руда считается оталькованной
          </span>
        </div>
        <div className={s.paramInputWrap}>
          <input
            type="number"
            min={0.5}
            max={99}
            step={0.5}
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
            aria-label="Порог оталькованности, %"
          />
          <span className={s.paramUnit}>%</span>
        </div>
      </div>

      {error && <div className={s.error}>{error}</div>}

      <div className={s.footer}>
        {busy && (
          <div className={s.uploadState}>
            {file && uploadPct !== null && uploadPct < 100 ? (
              <>
                <span>
                  загрузка файла… {fmtNum(uploadPct, 0)}% из {fmtBytes(file.size)}
                </span>
                <ProgressBar percent={uploadPct} />
              </>
            ) : (
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Spinner size={13} /> постановка в очередь…
              </span>
            )}
          </div>
        )}
        <button className="btn btn-ghost" onClick={onClose} disabled={busy} type="button">
          Отмена
        </button>
        <button className="btn btn-primary" onClick={() => void submit()} disabled={!canSubmit} type="button">
          Запустить анализ
        </button>
      </div>
    </Modal>
  )
}
