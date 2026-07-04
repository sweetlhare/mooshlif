import { useEffect, useId, useRef, type CSSProperties, type ReactNode } from 'react'
import s from './ui.module.css'

/* ---------- Светодиод ---------- */

export function Led({ color, pulse = false }: { color: string; pulse?: boolean }) {
  return (
    <span
      className={`${s.led} ${pulse ? s.ledPulse : ''}`}
      style={{ '--led-color': color } as CSSProperties}
      aria-hidden
    />
  )
}

/* ---------- Спиннер ---------- */

export function Spinner({ size = 18 }: { size?: number }) {
  return (
    <span
      className={s.spinner}
      style={{ width: size, height: size }}
      role="progressbar"
      aria-label="Загрузка"
    />
  )
}

/* ---------- Скелетон ---------- */

export function Skeleton({
  width,
  height,
  style,
}: {
  width?: number | string
  height?: number | string
  style?: CSSProperties
}) {
  return <div className={s.skeleton} style={{ width, height, ...style }} aria-hidden />
}

/* ---------- Прогресс-бар ---------- */

export function ProgressBar({
  percent,
  indeterminate = false,
  color,
}: {
  percent: number
  indeterminate?: boolean
  color?: string
}) {
  return (
    <div className={s.track}>
      <div
        className={`${s.fill} ${indeterminate ? s.fillIndeterminate : ''}`}
        style={{
          width: `${Math.min(100, Math.max(0, percent))}%`,
          ...(color ? { background: color } : null),
        }}
      />
    </div>
  )
}

/* ---------- Модальное окно ---------- */

export function Modal({
  title,
  width = 560,
  onClose,
  children,
}: {
  title: ReactNode
  width?: number
  onClose: () => void
  children: ReactNode
}) {
  const modalRef = useRef<HTMLDivElement>(null)
  const titleId = useId()

  useEffect(() => {
    const prevFocus = document.activeElement as HTMLElement | null
    const focusables = () =>
      modalRef.current
        ? Array.from(
            modalRef.current.querySelectorAll<HTMLElement>(
              'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
            ),
          ).filter((el) => !el.hasAttribute('disabled') && el.offsetParent !== null)
        : []
    focusables()[0]?.focus()

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
        return
      }
      if (e.key === 'Tab') {
        const f = focusables()
        if (f.length === 0) return
        const first = f[0]
        const last = f[f.length - 1]
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault()
          first.focus()
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      prevFocus?.focus?.()
    }
  }, [onClose])

  return (
    <div className={s.overlay} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div ref={modalRef} className={s.modal} style={{ maxWidth: width }} role="dialog" aria-modal aria-labelledby={titleId}>
        <div className={s.modalHead}>
          <div className={s.modalTitle} id={titleId}>{title}</div>
          <button className={s.modalClose} onClick={onClose} aria-label="Закрыть" type="button">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M2 2l10 10M12 2L2 12" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <div className={s.modalBody}>{children}</div>
      </div>
    </div>
  )
}
