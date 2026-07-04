import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { Health } from '../api/types'

interface HealthState {
  health: Health | null
  offline: boolean
}

/** Статус системы: /api/health, опрос каждые 15 с. */
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ health: null, offline: false })

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        const health = await api.health()
        if (!cancelled) setState({ health, offline: false })
      } catch {
        if (!cancelled) setState((s) => ({ health: s.health, offline: true }))
      }
    }
    void check()
    const t = window.setInterval(() => void check(), 15_000)
    return () => {
      cancelled = true
      window.clearInterval(t)
    }
  }, [])

  return state
}
