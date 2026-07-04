import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import { urls } from '../api/client'
import { Spinner } from './ui'
import s from './OsdViewer.module.css'

export interface LayerState {
  imageVisible: boolean
  phasesVisible: boolean
  /** 0..1 */
  phasesOpacity: number
}

/**
 * Deep-zoom просмотр: слой 0 — исходник (image.dzi), слой 1 — маска фаз
 * (phases.dzi, PNG с альфой) с регулируемой прозрачностью.
 */
export function OsdViewer({
  analysisId,
  layers,
}: {
  analysisId: string
  layers: LayerState
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null)
  const layersRef = useRef(layers)
  layersRef.current = layers
  const [state, setState] = useState<'loading' | 'ready' | 'failed'>('loading')

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    setState('loading')

    const viewer = OpenSeadragon({
      element: host,
      showNavigationControl: false,
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      navigatorSizeRatio: 0.16,
      navigatorDisplayRegionColor: '#e3a63b',
      animationTime: 0.55,
      springStiffness: 7.5,
      maxZoomPixelRatio: 2.5,
      minZoomImageRatio: 0.75,
      visibilityRatio: 0.4,
      timeout: 90_000,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
    } as OpenSeadragon.Options)

    viewerRef.current = viewer

    const applyLayers = () => {
      const world = viewer.world
      const L = layersRef.current
      const image = world.getItemAt(0)
      const phases = world.getItemAt(1)
      // Жёстко совмещаем слой маски с базовым: одинаковые bounds, даже если
      // пиксельные размеры пирамид различаются (маска — в рабочем масштабе,
      // вдвое меньше исходника). Без этого OSD может показать маску со сдвигом.
      if (image && phases) {
        const b = image.getBounds()
        const pb = phases.getBounds()
        if (pb.x !== b.x || pb.y !== b.y || Math.abs(pb.width - b.width) > 1e-9) {
          phases.setPosition(new OpenSeadragon.Point(b.x, b.y), true)
          phases.setWidth(b.width, true)
        }
      }
      if (image) image.setOpacity(L.imageVisible ? 1 : 0)
      if (phases) phases.setOpacity(L.phasesVisible ? L.phasesOpacity : 0)
    }
    // Сохраняем на инстансе, чтобы дёргать из эффекта пропсов.
    ;(viewer as unknown as { __applyLayers?: () => void }).__applyLayers = applyLayers

    // «open» не срабатывает при добавлении слоёв через addTiledImage —
    // готовность отслеживаем по первому add-item, ошибки — по колбэкам.
    viewer.world.addHandler('add-item', () => {
      applyLayers()
      setState((prev) => (prev === 'loading' ? 'ready' : prev))
    })

    viewer.addTiledImage({
      tileSource: urls.dzi(analysisId, 'image'),
      index: 0,
      error: () => setState('failed'),
    })
    viewer.addTiledImage({
      tileSource: urls.dzi(analysisId, 'phases'),
      index: 1,
      opacity: layersRef.current.phasesVisible ? layersRef.current.phasesOpacity : 0,
      // Маска может отсутствовать — исходник при этом должен остаться доступным.
      error: () => undefined,
    })

    return () => {
      viewer.destroy()
      viewerRef.current = null
    }
  }, [analysisId])

  useEffect(() => {
    const viewer = viewerRef.current as unknown as { __applyLayers?: () => void } | null
    viewer?.__applyLayers?.()
  }, [layers.imageVisible, layers.phasesVisible, layers.phasesOpacity])

  const zoom = (factor: number) => {
    const vp = viewerRef.current?.viewport
    if (!vp) return
    vp.zoomBy(factor)
    vp.applyConstraints()
  }

  return (
    <div className={`${s.root} osd-root`}>
      <div ref={hostRef} className={s.canvasHost} />

      {state === 'loading' && (
        <div className={s.veil}>
          <Spinner size={26} />
          <span>загрузка пирамиды тайлов…</span>
        </div>
      )}
      {state === 'failed' && (
        <div className={`${s.veil} ${s.failed}`}>
          <span>Не удалось загрузить тайлы изображения</span>
        </div>
      )}

      <div className={s.zoomControls}>
        <button className={s.zoomBtn} onClick={() => zoom(1.45)} title="Приблизить" type="button">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
            <path d="M7 2v10M2 7h10" strokeLinecap="round" />
          </svg>
        </button>
        <button className={s.zoomBtn} onClick={() => zoom(1 / 1.45)} title="Отдалить" type="button">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
            <path d="M2 7h10" strokeLinecap="round" />
          </svg>
        </button>
        <button
          className={s.zoomBtn}
          onClick={() => viewerRef.current?.viewport.goHome()}
          title="Показать весь снимок"
          type="button"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
            <rect x="2.5" y="2.5" width="11" height="11" rx="1.5" />
            <circle cx="8" cy="8" r="1.6" fill="currentColor" stroke="none" />
          </svg>
        </button>
      </div>
    </div>
  )
}
