import type { AnalysisStatus } from '../api/types'

/** Форматирование числа в русской локали (десятичная запятая, неразрывные разряды). */
export function fmtNum(x: number, digits = 1): string {
  return x.toLocaleString('ru-RU', {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  })
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  if (x == null || !Number.isFinite(x)) return '—'
  return `${fmtNum(x, digits)}%`
}

export function fmtElapsed(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return '—'
  if (s < 60) return `${fmtNum(s, s < 10 ? 1 : 0)} с`
  const m = Math.floor(s / 60)
  const rest = Math.round(s % 60)
  return `${m} мин ${rest} с`
}

export function fmtDateTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export const STATUS_LABEL: Record<AnalysisStatus, string> = {
  queued: 'В очереди',
  running: 'Обработка',
  done: 'Готово',
  error: 'Ошибка',
}

export const STATUS_COLOR: Record<AnalysisStatus, string> = {
  queued: 'var(--queue)',
  running: 'var(--run)',
  done: 'var(--ok)',
  error: 'var(--err)',
}

/**
 * Цвет класса руды. Контракт фиксирует только «оталькованная», поэтому набор
 * правил по ключевым словам + стабильный fallback-акцент для незнакомых классов.
 */
const CLASS_RULES: Array<[RegExp, string]> = [
  [/тальк/i, 'var(--phase-talc)'],
  [/труднообогат|тонк/i, 'var(--phase-fine, #f0564f)'],
  [/сплошн|массивн|богат/i, '#f0564f'],
  [/прожилк/i, '#e8873c'],
  [/вкраплен/i, 'var(--phase-ordinary)'],
  [/рядов|обычн|норм|бедн|неотальк/i, '#3ed07f'],
]

export function oreClassColor(cls: string | null | undefined): string {
  if (!cls) return 'var(--text-2)'
  for (const [re, color] of CLASS_RULES) {
    if (re.test(cls)) return color
  }
  return 'var(--accent)'
}

/** Подписи для морфометрических признаков (result.features). */
export const FEATURE_LABELS: Record<string, string> = {
  thick_med: 'Медианная толщина, px',
  n_comp_per_ka: 'Компонент на 1000 px²',
  perim_per_area: 'Периметр / площадь',
  inclusion_frac: 'Доля включений',
}

/** Цвет чипа веса фактора в объяснении решения. */
export function weightColor(weight: string): string {
  const w = weight.toLowerCase()
  if (w.includes('решающ')) return 'var(--accent)'
  if (w.includes('высок')) return 'var(--run)'
  if (w.includes('средн')) return 'var(--text-1)'
  return 'var(--text-2)'
}
