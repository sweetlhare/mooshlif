import { useCallback, useState } from 'react'
import { api, errorMessage } from '../api/client'

export interface UploadJob {
  tempId: string
  name: string
  size: number
  isFile: boolean
  pct: number
  phase: 'uploading' | 'error'
  error?: string
}

export interface UploadRequest {
  file?: File
  serverPath?: string
  name: string
  size: number
  isFile: boolean
}

let counter = 0

/**
 * Фоновые загрузки: живут на уровне App, поэтому переживают закрытие модалки и
 * переходы между страницами. Файлы грузятся последовательно (бэкенд всё равно
 * обрабатывает по одному); по завершении загрузки анализ уже в очереди на
 * бэкенде, а строка-заглушка убирается и список перезагружается (doneTick).
 */
export function useUploads() {
  const [uploads, setUploads] = useState<UploadJob[]>([])
  const [doneTick, setDoneTick] = useState(0)

  const startUploads = useCallback((items: UploadRequest[], threshold: number) => {
    const jobs = items.map((req) => ({
      req,
      job: {
        tempId: `u${(counter += 1)}`,
        name: req.name,
        size: req.size,
        isFile: req.isFile,
        pct: 0,
        phase: 'uploading' as const,
      } satisfies UploadJob,
    }))
    setUploads((prev) => [...jobs.map((j) => j.job), ...prev])

    void (async () => {
      for (const { req, job } of jobs) {
        try {
          await api.createAnalysis({
            file: req.file,
            serverPath: req.serverPath,
            talcThresholdPct: threshold,
            onUploadProgress: req.isFile
              ? (fr) =>
                  setUploads((prev) =>
                    prev.map((u) =>
                      u.tempId === job.tempId ? { ...u, pct: Math.min(100, Math.round(fr * 100)) } : u,
                    ),
                  )
              : undefined,
          })
          setUploads((prev) => prev.filter((u) => u.tempId !== job.tempId))
          setDoneTick((t) => t + 1)
        } catch (e) {
          setUploads((prev) =>
            prev.map((u) =>
              u.tempId === job.tempId ? { ...u, phase: 'error', error: errorMessage(e) } : u,
            ),
          )
        }
      }
    })()
  }, [])

  const dismiss = useCallback(
    (tempId: string) => setUploads((prev) => prev.filter((u) => u.tempId !== tempId)),
    [],
  )

  return { uploads, startUploads, dismiss, doneTick }
}
