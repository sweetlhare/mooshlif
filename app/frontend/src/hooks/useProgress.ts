import { useEffect, useRef, useState } from 'react'
import { urls } from '../api/client'
import type { ProgressEvent as AnalysisProgressEvent } from '../api/types'

/**
 * Живой прогресс анализа через SSE (/api/analyses/{id}/events).
 * `id === null` — подписка выключена. `onFinal` вызывается один раз при
 * финальном событии (done/error) или обрыве потока — сигнал перезагрузить данные.
 */
export function useProgress(
  id: string | null,
  onFinal?: (stage: string) => void,
): AnalysisProgressEvent | null {
  const [progress, setProgress] = useState<AnalysisProgressEvent | null>(null)
  const onFinalRef = useRef(onFinal)
  onFinalRef.current = onFinal

  useEffect(() => {
    if (!id) {
      setProgress(null)
      return
    }
    let finished = false
    const es = new EventSource(urls.events(id))

    es.onmessage = (e: MessageEvent<string>) => {
      let data: AnalysisProgressEvent
      try {
        data = JSON.parse(e.data) as AnalysisProgressEvent
      } catch {
        return
      }
      setProgress(data)
      if (data.stage === 'done' || data.stage === 'error' || data.percent >= 100) {
        finished = true
        es.close()
        onFinalRef.current?.(data.stage)
      }
    }

    es.onerror = () => {
      // CONNECTING — браузер сам переподключится; CLOSED — поток оборван насовсем.
      if (es.readyState === EventSource.CLOSED && !finished) {
        finished = true
        onFinalRef.current?.('interrupted')
      }
    }

    return () => es.close()
  }, [id])

  return progress
}
