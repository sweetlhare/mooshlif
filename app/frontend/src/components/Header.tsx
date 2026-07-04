import type { Health } from '../api/types'
import { Led } from './ui'
import s from './Header.module.css'

function ReticleMark() {
  return (
    <svg className={s.mark} width="26" height="26" viewBox="0 0 32 32" fill="none" aria-hidden>
      <circle cx="16" cy="16" r="9.5" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M16 2.5v6M16 23.5v6M2.5 16h6M23.5 16h6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <circle cx="16" cy="16" r="2.2" fill="currentColor" />
    </svg>
  )
}

export function Header({
  health,
  offline,
  onHome,
  onFeedback,
  active,
}: {
  health: Health | null
  offline: boolean
  onHome: () => void
  onFeedback?: () => void
  active?: string
}) {
  const models = health ? Object.entries(health.models) : []

  return (
    <header className={s.header}>
      <button className={s.brand} onClick={onHome} type="button" title="К списку анализов">
        <ReticleMark />
        <span className={s.wordmarkWrap}>
          <span className={s.wordmark}>
            ШЛИФ<em>-СКАН</em>
          </span>
          <span className={s.tagline}>панорамная минераграфия</span>
        </span>
      </button>

      <nav className={s.nav}>
        <button
          type="button"
          className={`${s.navLink} ${active === 'list' || active === 'analysis' ? s.navActive : ''}`}
          onClick={onHome}
        >
          Анализы
        </button>
        {onFeedback && (
          <button
            type="button"
            className={`${s.navLink} ${active === 'feedback' || active === 'annotate' ? s.navActive : ''}`}
            onClick={onFeedback}
            title="Снимки, отправленные на доработку и разметку для дообучения"
          >
            На доработку
          </button>
        )}
      </nav>

      <div className={s.health} title="Статус системы (/api/health)">
        {offline ? (
          <>
            <Led color="var(--err)" pulse />
            <span className={s.healthOffline}>НЕТ СВЯЗИ С СЕРВЕРОМ</span>
          </>
        ) : health ? (
          <>
            <Led color={health.status === 'ok' ? 'var(--ok)' : 'var(--err)'} />
            <span className={s.healthDevice}>{health.device}</span>
            {models.map(([name, ver]) => (
              <span key={name} style={{ display: 'inline-flex', alignItems: 'center', gap: 10 }}>
                <span className={s.healthSep} />
                <span>
                  {name} {ver}
                </span>
              </span>
            ))}
          </>
        ) : (
          <>
            <Led color="var(--queue)" pulse />
            <span>подключение…</span>
          </>
        )}
      </div>
    </header>
  )
}
