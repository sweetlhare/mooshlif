import { useEffect, useState } from 'react'
import { api, errorMessage, type FeedbackItem } from '../api/client'
import s from './FeedbackPage.module.css'

export function FeedbackPage({
  onOpen,
  onAnnotate,
  onBack,
}: {
  onOpen: (id: string) => void
  onAnnotate: (id: string) => void
  onBack: () => void
}) {
  const [items, setItems] = useState<FeedbackItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = () => {
    setError(null)
    api
      .listFeedback()
      .then(setItems)
      .catch((e) => setError(errorMessage(e)))
  }
  useEffect(load, [])

  const unflag = async (id: string) => {
    try {
      await api.unflag(id)
      load()
    } catch (e) {
      setError(errorMessage(e))
    }
  }

  return (
    <div className={s.page}>
      <div className={s.head}>
        <div>
          <h1 className={s.title}>Снимки на доработку</h1>
          <p className={s.sub}>
            Очередь снимков, помеченных технологом как проблемные. Разметьте фазы
            вручную — размеченные примеры пойдут в дообучение моделей.
          </p>
        </div>
        <button className="btn btn-ghost" onClick={onBack} type="button">
          К анализам
        </button>
      </div>

      {error && <div className={s.error}>{error}</div>}

      {items === null ? (
        <div className={s.muted}>Загрузка…</div>
      ) : items.length === 0 ? (
        <div className={s.empty}>
          <div className={s.emptyMark}>✓</div>
          <div>
            Очередь пуста. Отправить снимок сюда можно кнопкой{' '}
            <b>«На доработку»</b> на странице анализа.
          </div>
        </div>
      ) : (
        <div className={s.list}>
          {items.map((it) => (
            <div key={it.id} className={s.card}>
              <div className={s.cardMain}>
                <div className={s.row1}>
                  <span className={s.fileName}>{it.file_name}</span>
                  {it.annotated ? (
                    <span className={`${s.badge} ${s.badgeDone}`}>размечено</span>
                  ) : (
                    <span className={`${s.badge} ${s.badgeTodo}`}>ждёт разметки</span>
                  )}
                </div>
                <div className={s.meta}>
                  {it.ore_class && <span>класс: <b>{it.ore_class}</b></span>}
                  {it.talc_pct != null && (
                    <span className="num">тальк {it.talc_pct.toFixed(1)}%</span>
                  )}
                  {it.flagged_at && (
                    <span className={s.metaDim}>{formatWhen(it.flagged_at)}</span>
                  )}
                </div>
                {it.reason && <div className={s.reason}>{reasonLabel(it.reason)}</div>}
                {it.note && <div className={s.note}>{it.note}</div>}
              </div>
              <div className={s.actions}>
                <button className="btn btn-primary" onClick={() => onAnnotate(it.id)} type="button">
                  {it.annotated ? 'Правка разметки' : 'Разметить'}
                </button>
                <button className="btn" onClick={() => onOpen(it.id)} type="button">
                  Анализ
                </button>
                <button className="btn btn-ghost btn-danger" onClick={() => unflag(it.id)} type="button">
                  Снять флаг
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function reasonLabel(r: string): string {
  const map: Record<string, string> = {
    wrong_class: 'Неверный класс руды',
    bad_talc: 'Ошибка по тальку',
    bad_segmentation: 'Плохая сегментация фаз',
    poor_quality: 'Низкое качество снимка',
    other: 'Другое',
  }
  return map[r] || r
}

function formatWhen(iso: string): string {
  try {
    return new Date(iso).toLocaleString('ru-RU', {
      day: '2-digit',
      month: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return iso
  }
}
