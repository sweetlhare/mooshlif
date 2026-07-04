import type {
  AnalysisDetail,
  AnalysisSummary,
  CreateAnalysisResponse,
  DemoImage,
  Health,
} from './types'

const API = '/api'

export class ApiError extends Error {
  readonly status: number | undefined

  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message
  if (e instanceof Error) return e.message
  return 'Неизвестная ошибка'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(API + path, init)
  } catch {
    throw new ApiError('Сервер недоступен')
  }
  if (!res.ok) {
    let detail = `Ошибка сервера (HTTP ${res.status})`
    try {
      const body: unknown = await res.json()
      if (body && typeof body === 'object' && 'detail' in body) {
        detail = String((body as { detail: unknown }).detail)
      }
    } catch {
      /* тело не JSON — оставляем общее сообщение */
    }
    throw new ApiError(detail, res.status)
  }
  return (await res.json()) as T
}

export interface CreateAnalysisOptions {
  file?: File
  serverPath?: string
  /** Порог оталькованности, % (params.talc.talc_ore_thr_pct) */
  talcThresholdPct?: number
  onUploadProgress?: (fraction: number) => void
}

/** POST /api/analyses через XHR — ради прогресса загрузки гигапиксельных файлов. */
function createAnalysis(opts: CreateAnalysisOptions): Promise<CreateAnalysisResponse> {
  return new Promise((resolve, reject) => {
    const fd = new FormData()
    if (opts.file) {
      fd.append('file', opts.file)
    } else if (opts.serverPath) {
      fd.append('server_path', opts.serverPath)
    } else {
      reject(new ApiError('Не выбран файл'))
      return
    }
    if (opts.talcThresholdPct != null && Number.isFinite(opts.talcThresholdPct)) {
      fd.append('params', JSON.stringify({ talc: { talc_ore_thr_pct: opts.talcThresholdPct } }))
    }

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API}/analyses`)
    xhr.responseType = 'json'
    if (opts.onUploadProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) opts.onUploadProgress?.(e.loaded / e.total)
      }
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as CreateAnalysisResponse)
      } else {
        const body = xhr.response as { detail?: unknown } | null
        const detail = body?.detail != null ? String(body.detail) : `Ошибка сервера (HTTP ${xhr.status})`
        reject(new ApiError(detail, xhr.status))
      }
    }
    xhr.onerror = () => reject(new ApiError('Сервер недоступен'))
    xhr.send(fd)
  })
}

const enc = encodeURIComponent

export const api = {
  health: () => request<Health>('/health'),
  demoImages: () => request<DemoImage[]>('/demo-images'),
  listAnalyses: () => request<AnalysisSummary[]>('/analyses'),
  getAnalysis: (id: string) => request<AnalysisDetail>(`/analyses/${enc(id)}`),
  createAnalysis,
  deleteAnalysis: async (id: string): Promise<void> => {
    let res: Response
    try {
      res = await fetch(`${API}/analyses/${enc(id)}`, { method: 'DELETE' })
    } catch {
      throw new ApiError('Сервер недоступен')
    }
    if (!res.ok) throw new ApiError(`Не удалось удалить (HTTP ${res.status})`, res.status)
  },
  // --- обратная связь / разметка для дообучения ---
  flag: (id: string, reason: string, note: string) =>
    request<{ ok: boolean }>(`/analyses/${enc(id)}/flag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason, note }),
    }),
  unflag: (id: string) =>
    request<{ ok: boolean }>(`/analyses/${enc(id)}/flag`, { method: 'DELETE' }),
  getFeedback: (id: string) => request<FeedbackState>(`/analyses/${enc(id)}/feedback`),
  listFeedback: () => request<FeedbackItem[]>('/feedback'),
  annotationPhases: () => request<AnnotationPhase[]>('/annotation/phases'),
  saveAnnotation: (id: string, payload: SaveAnnotation) =>
    request<{ ok: boolean }>(`/analyses/${enc(id)}/annotation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
}

export interface FeedbackState {
  flagged: boolean
  reason: string
  note: string
  annotated: boolean
  flagged_at?: string
  annotated_at?: string
}
export interface FeedbackItem extends AnalysisSummary {
  reason: string
  note: string
  flagged_at?: string
}
export interface AnnotationPhase {
  key: string
  label: string
  color: string
}
export interface SaveAnnotation {
  image: string
  phases: AnnotationPhase[]
  width: number
  height: number
}

export const urls = {
  dzi: (id: string, layer: 'image' | 'phases') =>
    `${API}/analyses/${encodeURIComponent(id)}/dzi/${layer}.dzi`,
  preview: (id: string) => `${API}/analyses/${encodeURIComponent(id)}/preview.jpg`,
  confidence: (id: string) => `${API}/analyses/${encodeURIComponent(id)}/confidence.jpg`,
  events: (id: string) => `${API}/analyses/${encodeURIComponent(id)}/events`,
  reportPdf: (id: string) => `${API}/analyses/${encodeURIComponent(id)}/report.pdf`,
  metricsCsv: (id: string) => `${API}/analyses/${encodeURIComponent(id)}/metrics.csv`,
  maskGeojson: (id: string) => `${API}/analyses/${encodeURIComponent(id)}/mask.geojson`,
  annotation: (id: string) => `${API}/analyses/${encodeURIComponent(id)}/annotation.png`,
  batchCsv: `${API}/export/batch.csv`,
}
