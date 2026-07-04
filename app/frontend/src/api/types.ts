/** Типы строго по docs/api_contract.md */

export type AnalysisStatus = 'queued' | 'running' | 'done' | 'error'

export interface AnalysisSummary {
  id: string
  file_name: string
  status: AnalysisStatus
  created_at: string
  ore_class: string | null
  talc_pct: number | null
  sulfide_total_pct: number | null
  elapsed_s: number | null
  flagged?: boolean
  annotated?: boolean
}

export interface AnalysisImageInfo {
  width: number
  height: number
  work_scale: number
}

export interface AnalysisMetrics {
  sulfide_total_pct: number
  ordinary_pct: number
  fine_pct: number
  ordinary_share: number
  fine_share: number
  talc_pct: number
  talc_ci: [number, number]
  gray_phase_pct: number
}

export interface Granulometry {
  ecd_p50: number
  ecd_p80: number
  /** [[диаметр, накопленная доля], ...] */
  curve: [number, number][]
}

export interface ExplanationItem {
  factor: string
  weight: string
}

export interface AnalysisResult {
  ore_class: string
  conclusion: string
  confidence: number
  metrics: AnalysisMetrics
  granulometry: Granulometry
  features: Record<string, number>
  explanation: ExplanationItem[]
  params: {
    talc_ore_thr_pct: number
    model_version: string
  }
}

export interface AnalysisDetail {
  id: string
  file_name: string
  status: AnalysisStatus
  created_at: string
  elapsed_s: number | null
  error: string | null
  image: AnalysisImageInfo | null
  result: AnalysisResult | null
}

export interface CreateAnalysisResponse {
  id: string
  status: string
}

export interface DemoImage {
  name: string
  server_path: string
  size: [number, number]
}

export interface Health {
  status: string
  models: Record<string, string>
  device: string
}

export interface ProgressEvent {
  stage: string
  percent: number
}
