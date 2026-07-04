import { useEffect, useRef, useState, type DragEvent } from 'react'
import { api } from '../api/client'
import type { DemoImage } from '../api/types'
import type { UploadRequest } from '../hooks/useUploads'
import { fmtNum } from '../lib/format'
import { Modal, Skeleton } from './ui'
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
  onSubmit,
}: {
  onClose: () => void
  onSubmit: (items: UploadRequest[], threshold: number) => void
}) {
  const [files, setFiles] = useState<File[]>([])
  const [selectedDemos, setSelectedDemos] = useState<DemoImage[]>([])
  const [demos, setDemos] = useState<DemoImage[] | null>(null)
  const [demosError, setDemosError] = useState(false)
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  const [dragOver, setDragOver] = useState(false)
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

  const addFiles = (list: FileList | File[]) => {
    const accepted: File[] = []
    const rejected: string[] = []
    for (const f of Array.from(list)) {
      if (isAcceptedFile(f)) accepted.push(f)
      else rejected.push(f.name)
    }
    setError(rejected.length ? `Пропущены (формат): ${rejected.join(', ')}` : null)
    if (accepted.length) {
      setFiles((prev) => {
        const key = (f: File) => `${f.name}:${f.size}`
        const seen = new Set(prev.map(key))
        return [...prev, ...accepted.filter((f) => !seen.has(key(f)))]
      })
    }
  }

  const removeFile = (idx: number) => setFiles((prev) => prev.filter((_, i) => i !== idx))

  const toggleDemo = (d: DemoImage) => {
    setError(null)
    setSelectedDemos((prev) =>
      prev.some((x) => x.server_path === d.server_path)
        ? prev.filter((x) => x.server_path !== d.server_path)
        : [...prev, d],
    )
  }
  const isDemoSelected = (d: DemoImage) =>
    selectedDemos.some((x) => x.server_path === d.server_path)

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files)
  }

  const parsedThreshold = (() => {
    const v = Number.parseFloat(threshold.replace(',', '.'))
    return Number.isFinite(v) && v > 0 && v < 100 ? v : null
  })()

  const total = files.length + selectedDemos.length
  const canSubmit = total > 0 && parsedThreshold !== null

  const submit = () => {
    if (!canSubmit || parsedThreshold === null) return
    const items: UploadRequest[] = [
      ...files.map((f) => ({ file: f, name: f.name, size: f.size, isFile: true })),
      ...selectedDemos.map((d) => ({ serverPath: d.server_path, name: d.name, size: 0, isFile: false })),
    ]
    onSubmit(items, parsedThreshold)
    onClose()
  }

  return (
    <Modal title="Новый анализ" onClose={onClose} width={580}>
      {/* --- Загрузка файлов (можно несколько) --- */}
      <div
        className={[
          s.dropzone,
          dragOver ? s.dropzoneActive : '',
          files.length ? s.dropzoneSelected : '',
        ].join(' ')}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT_EXT.join(',')}
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files?.length) addFiles(e.target.files)
            e.target.value = ''
          }}
        />
        {files.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, width: '100%' }}>
            {files.map((f, i) => (
              <div key={`${f.name}:${f.size}:${i}`} className={s.selectedFile}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
                  <path d="M13.5 4.5l-7 7L2.5 8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span className={s.selectedName}>{f.name}</span>
                <span style={{ color: 'var(--text-2)' }}>{fmtBytes(f.size)}</span>
                <button
                  className={s.clearFile}
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile(i)
                  }}
                  title="Убрать файл"
                  type="button"
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
                    <path d="M2 2l8 8M10 2l-8 8" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
            ))}
            <div className={s.dropHint} style={{ paddingTop: 2 }}>
              + перетащите ещё или{' '}
              <button
                type="button"
                className={s.linkBtn}
                onClick={(e) => {
                  e.stopPropagation()
                  inputRef.current?.click()
                }}
              >
                выберите файлы
              </button>
            </div>
          </div>
        ) : (
          <>
            <svg className={s.dropIcon} width="34" height="34" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
              <path d="M16 21V7m0 0l-5.5 5.5M16 7l5.5 5.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M5 21.5v3a2 2 0 002 2h18a2 2 0 002-2v-3" strokeLinecap="round" />
            </svg>
            <div className={s.dropTitle}>
              Перетащите панорамы шлифов или{' '}
              <button
                type="button"
                className={s.linkBtn}
                onClick={(e) => {
                  e.stopPropagation()
                  inputRef.current?.click()
                }}
              >
                выберите файлы
              </button>
            </div>
            <div className={s.dropHint}>
              Можно сразу несколько · TIFF, PNG, JPEG, BMP — вплоть до гигапикселя
            </div>
          </>
        )}
      </div>

      {/* --- Демо-набор (мультивыбор) --- */}
      <div className={s.divider}>
        <span className="microlabel">или демо-набор — можно отметить несколько</span>
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
              className={`${s.demoItem} ${isDemoSelected(d) ? s.demoItemActive : ''}`}
              onClick={() => toggleDemo(d)}
              aria-pressed={isDemoSelected(d)}
              type="button"
            >
              <span className={s.demoName}>
                {isDemoSelected(d) ? '☑ ' : '☐ '}
                {d.name}
              </span>
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
        <span className={s.footerHint}>
          Загрузка и обработка идут в фоне — можно продолжать работу.
        </span>
        <button className="btn btn-ghost" onClick={onClose} type="button">
          Отмена
        </button>
        <button className="btn btn-primary" onClick={submit} disabled={!canSubmit} type="button">
          {total > 1 ? `В очередь (${total})` : 'В очередь'}
        </button>
      </div>
    </Modal>
  )
}
