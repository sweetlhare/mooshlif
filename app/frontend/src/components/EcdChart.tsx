import { useMemo } from 'react'
import type { Granulometry } from '../api/types'
import { fmtNum } from '../lib/format'

const W = 320
const H = 150
const PAD = { top: 12, right: 12, bottom: 24, left: 34 }

/**
 * Кумулятивная кривая ECD (гранулометрия сульфидов) с маркерами P50/P80.
 * Ось X — логарифмическая при большом динамическом диапазоне диаметров.
 */
export function EcdChart({ granulometry }: { granulometry: Granulometry }) {
  const model = useMemo(() => {
    const raw = (granulometry.curve ?? [])
      .filter(
        (p): p is [number, number] =>
          Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]) && p[0] > 0,
      )
      .sort((a, b) => a[0] - b[0])
    if (raw.length < 2) return null

    // Накопленная доля может прийти как 0..1 или 0..100 — нормализуем в 0..100.
    const maxFrac = Math.max(...raw.map(([, f]) => f))
    const scale = maxFrac <= 1.5 ? 100 : 1
    const pts = raw.map(([d, f]) => [d, Math.min(100, f * scale)] as const)

    const dMin = pts[0][0]
    const dMax = pts[pts.length - 1][0]
    const useLog = dMax / dMin > 20

    const tx = (d: number) => {
      const t = useLog
        ? (Math.log10(d) - Math.log10(dMin)) / (Math.log10(dMax) - Math.log10(dMin))
        : (d - dMin) / (dMax - dMin)
      return PAD.left + t * (W - PAD.left - PAD.right)
    }
    const ty = (frac: number) => PAD.top + (1 - frac / 100) * (H - PAD.top - PAD.bottom)

    const line = pts.map(([d, f], i) => `${i === 0 ? 'M' : 'L'}${tx(d).toFixed(1)},${ty(f).toFixed(1)}`).join(' ')
    const area = `${line} L${tx(dMax).toFixed(1)},${ty(0).toFixed(1)} L${tx(dMin).toFixed(1)},${ty(0).toFixed(1)} Z`

    const markers = [
      { label: 'P50', value: granulometry.ecd_p50 },
      { label: 'P80', value: granulometry.ecd_p80 },
    ].filter((m) => Number.isFinite(m.value) && m.value >= dMin && m.value <= dMax)

    return { tx, ty, line, area, dMin, dMax, markers }
  }, [granulometry])

  if (!model) {
    return (
      <div style={{ color: 'var(--text-2)', fontSize: 12, padding: '8px 0' }}>
        Недостаточно данных для кривой ECD
      </div>
    )
  }

  const { tx, ty, line, area, dMin, dMax, markers } = model
  const gridY = [25, 50, 75]

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', height: 'auto', display: 'block' }}
      role="img"
      aria-label="Кумулятивная кривая ECD"
    >
      <defs>
        <linearGradient id="ecdArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#e3a63b" stopOpacity="0.28" />
          <stop offset="100%" stopColor="#e3a63b" stopOpacity="0.02" />
        </linearGradient>
      </defs>

      {/* сетка */}
      {gridY.map((g) => (
        <g key={g}>
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={ty(g)}
            y2={ty(g)}
            stroke="rgba(151,168,194,0.12)"
            strokeDasharray="3 4"
          />
          <text x={PAD.left - 6} y={ty(g) + 3} textAnchor="end" fontSize={8.5} fill="var(--text-2)" fontFamily="var(--font-mono)">
            {g}%
          </text>
        </g>
      ))}
      <line x1={PAD.left} x2={W - PAD.right} y1={ty(0)} y2={ty(0)} stroke="rgba(151,168,194,0.3)" />
      <text x={PAD.left - 6} y={ty(0) + 3} textAnchor="end" fontSize={8.5} fill="var(--text-2)" fontFamily="var(--font-mono)">
        0
      </text>
      <text x={PAD.left - 6} y={ty(100) + 3} textAnchor="end" fontSize={8.5} fill="var(--text-2)" fontFamily="var(--font-mono)">
        100%
      </text>

      {/* кривая */}
      <path d={area} fill="url(#ecdArea)" />
      <path d={line} fill="none" stroke="var(--accent)" strokeWidth={1.8} strokeLinejoin="round" />

      {/* маркеры P50 / P80 */}
      {markers.map((m) => (
        <g key={m.label}>
          <line
            x1={tx(m.value)}
            x2={tx(m.value)}
            y1={PAD.top}
            y2={ty(0)}
            stroke="rgba(79,142,247,0.65)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          <text
            x={tx(m.value)}
            y={PAD.top - 3}
            textAnchor="middle"
            fontSize={8.5}
            fontWeight={600}
            fill="#7aa7f9"
            fontFamily="var(--font-mono)"
          >
            {m.label}
          </text>
        </g>
      ))}

      {/* подписи оси X */}
      <text x={PAD.left} y={H - 8} fontSize={8.5} fill="var(--text-2)" fontFamily="var(--font-mono)">
        {fmtNum(dMin, 1)}
      </text>
      <text x={W - PAD.right} y={H - 8} textAnchor="end" fontSize={8.5} fill="var(--text-2)" fontFamily="var(--font-mono)">
        {fmtNum(dMax, 0)} мкм
      </text>
    </svg>
  )
}
