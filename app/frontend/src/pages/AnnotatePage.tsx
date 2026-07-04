import { useEffect, useRef, useState } from 'react'
import { api, errorMessage, urls, type AnnotationPhase } from '../api/client'
import s from './AnnotatePage.module.css'

type Tool = 'brush' | 'erase' | 'polygon'
type Pt = { x: number; y: number }
const MIN_S = 0.05
const MAX_S = 40

export function AnnotatePage({ id, onBack }: { id: string; onBack: () => void }) {
  const stageRef = useRef<HTMLDivElement>(null)
  const paneRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const previewRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const view = useRef({ scale: 1, tx: 0, ty: 0 })
  const painting = useRef(false)
  const panning = useRef(false)
  const space = useRef(false)
  const last = useRef<Pt | null>(null)
  const poly = useRef<Pt[]>([])
  const toolRef = useRef<Tool>('brush')
  // Всегда актуальные обработчики полигона — чтобы keydown-эффект ([] deps) не
  // ловил замыкание первого рендера (иначе Enter-замыкание красит фоллбэком).
  const closePolyRef = useRef<() => void>(() => undefined)
  const cancelPolyRef = useRef<() => void>(() => undefined)

  const [phases, setPhases] = useState<AnnotationPhase[]>([])
  const [phase, setPhase] = useState<string>('talc')
  const [tool, setTool] = useState<Tool>('brush')
  const [brush, setBrush] = useState(28)
  const [zoomPct, setZoomPct] = useState(100)
  const [grabbing, setGrabbing] = useState(false)
  const [polyLen, setPolyLen] = useState(0)
  const [ready, setReady] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const setToolBoth = (t: Tool) => {
    toolRef.current = t
    setTool(t)
    if (t !== 'polygon') cancelPoly()
  }

  useEffect(() => {
    let cancelled = false
    api
      .annotationPhases()
      .then((p) => {
        if (cancelled) return
        setPhases(p)
        if (p.length) setPhase(p[0].key)
      })
      .catch(() => !cancelled && setErr('Не удалось загрузить палитру фаз'))
    return () => {
      cancelled = true
    }
  }, [])

  // Предупреждаем о несохранённой разметке при закрытии/перезагрузке вкладки.
  useEffect(() => {
    if (!dirty) return
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [dirty])

  const guardedBack = () => {
    if (dirty && !window.confirm('Разметка не сохранена. Выйти без сохранения?')) return
    onBack()
  }

  const applyView = () => {
    const { scale, tx, ty } = view.current
    if (paneRef.current) paneRef.current.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`
    setZoomPct(Math.round(scale * 100))
    redrawPreview()
  }

  const fit = () => {
    const st = stageRef.current
    const cv = canvasRef.current
    if (!st || !cv || !cv.width) return
    const scale = Math.min(st.clientWidth / cv.width, st.clientHeight / cv.height) * 0.94
    view.current = {
      scale,
      tx: (st.clientWidth - cv.width * scale) / 2,
      ty: (st.clientHeight - cv.height * scale) / 2,
    }
    applyView()
  }

  const zoomAt = (cx: number, cy: number, factor: number) => {
    const { scale, tx, ty } = view.current
    const ns = Math.min(MAX_S, Math.max(MIN_S, scale * factor))
    const k = ns / scale
    view.current = { scale: ns, tx: cx - (cx - tx) * k, ty: cy - (cy - ty) * k }
    applyView()
  }

  const color = () => phases.find((p) => p.key === phase)?.color ?? '#2f6fd6'

  const onImgLoad = () => {
    const img = imgRef.current
    const cv = canvasRef.current
    const pv = previewRef.current
    const pane = paneRef.current
    if (!img || !cv || !pv || !pane) return
    cv.width = pv.width = img.naturalWidth
    cv.height = pv.height = img.naturalHeight
    pane.style.width = `${img.naturalWidth}px`
    pane.style.height = `${img.naturalHeight}px`
    setBrush(Math.max(16, Math.round(img.naturalWidth / 55)))
    const prev = new Image()
    prev.crossOrigin = 'anonymous'
    prev.onload = () => cv.getContext('2d')?.drawImage(prev, 0, 0, cv.width, cv.height)
    prev.src = `${urls.annotation(id)}?t=${Date.now()}`
    setReady(true)
    fit()
  }

  // колесо: без ctrl — панорама (тачпад-скролл), ctrl/щипок — масштаб к курсору
  useEffect(() => {
    const st = stageRef.current
    if (!st) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault() // страница не листается ни при каком раскладе
      const r = st.getBoundingClientRect()
      const cx = e.clientX - r.left
      const cy = e.clientY - r.top
      // Тачпад: два пальца (есть deltaX или мелкий вертикальный шаг) → панорама;
      //         щипок приходит с ctrlKey → масштаб. Мышь: крупный вертикальный
      //         шаг колеса → масштаб к курсору (без «листания» вверх-вниз).
      const trackpadPan =
        !e.ctrlKey && !e.metaKey && (e.deltaX !== 0 || Math.abs(e.deltaY) < 40)
      if (trackpadPan) {
        view.current.tx -= e.deltaX
        view.current.ty -= e.deltaY
        applyView()
      } else {
        zoomAt(cx, cy, e.deltaY < 0 ? 1.12 : 1 / 1.12)
      }
    }
    st.addEventListener('wheel', onWheel, { passive: false })
    return () => st.removeEventListener('wheel', onWheel)
  }, [])

  // Пробел = панорама; Escape = отмена полигона; Enter = замкнуть
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !space.current) {
        space.current = true
        setGrabbing(true)
        e.preventDefault()
      } else if (e.code === 'Escape') {
        cancelPolyRef.current()
      } else if (e.code === 'Enter' && toolRef.current === 'polygon') {
        closePolyRef.current()
      }
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        space.current = false
        setGrabbing(false)
      }
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  const pt = (e: { clientX: number; clientY: number }): Pt => {
    const cv = canvasRef.current!
    const rect = cv.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) / rect.width) * cv.width,
      y: ((e.clientY - rect.top) / rect.height) * cv.height,
    }
  }

  // ---- кисть / ластик ----
  const stroke = (a: Pt, b: Pt) => {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    ctx.lineWidth = brush
    if (toolRef.current === 'erase') {
      ctx.globalCompositeOperation = 'destination-out'
      ctx.strokeStyle = 'rgba(0,0,0,1)'
    } else {
      ctx.globalCompositeOperation = 'source-over'
      ctx.strokeStyle = color()
    }
    ctx.beginPath()
    ctx.moveTo(a.x, a.y)
    ctx.lineTo(b.x, b.y)
    ctx.stroke()
  }

  // ---- полигон ----
  const redrawPreview = () => {
    const pv = previewRef.current
    const ctx = pv?.getContext('2d')
    if (!pv || !ctx) return
    ctx.clearRect(0, 0, pv.width, pv.height)
    const pts = poly.current
    if (!pts.length) return
    const sc = view.current.scale || 1
    ctx.lineWidth = 2 / sc
    ctx.strokeStyle = color()
    ctx.fillStyle = color()
    ctx.beginPath()
    ctx.moveTo(pts[0].x, pts[0].y)
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
    ctx.stroke()
    const r = 4 / sc
    for (const p of pts) {
      ctx.beginPath()
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  const closePoly = () => {
    const pts = poly.current
    const ctx = canvasRef.current?.getContext('2d')
    if (ctx && pts.length >= 3) {
      ctx.globalCompositeOperation = 'source-over'
      ctx.fillStyle = color()
      ctx.beginPath()
      ctx.moveTo(pts[0].x, pts[0].y)
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
      ctx.closePath()
      ctx.fill()
      setDirty(true)
    }
    poly.current = []
    setPolyLen(0)
    redrawPreview()
  }

  const cancelPoly = () => {
    poly.current = []
    setPolyLen(0)
    redrawPreview()
  }
  closePolyRef.current = closePoly
  cancelPolyRef.current = cancelPoly

  // ---- pointer ----
  const onDown = (e: React.PointerEvent) => {
    if (!ready) return
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    if (e.button === 1 || space.current) {
      panning.current = true
      setGrabbing(true)
      return
    }
    if (e.button !== 0) return
    if (toolRef.current === 'polygon') {
      const p = pt(e)
      const pts = poly.current
      const near = pts.length >= 3 && Math.hypot(p.x - pts[0].x, p.y - pts[0].y) < 10 / view.current.scale
      if (near) {
        closePoly()
      } else {
        pts.push(p)
        setPolyLen(pts.length)
        redrawPreview()
      }
      return
    }
    painting.current = true
    const p = pt(e)
    last.current = p
    stroke(p, p)
    setDirty(true)
  }

  const onMove = (e: React.PointerEvent) => {
    if (panning.current) {
      view.current.tx += e.movementX
      view.current.ty += e.movementY
      applyView()
      return
    }
    if (toolRef.current === 'polygon' && poly.current.length) {
      // резиновая линия к курсору
      const pv = previewRef.current
      const ctx = pv?.getContext('2d')
      if (pv && ctx) {
        redrawPreview()
        const p = pt(e)
        const pts = poly.current
        ctx.lineWidth = 2 / (view.current.scale || 1)
        ctx.strokeStyle = color()
        ctx.setLineDash([6 / view.current.scale, 4 / view.current.scale])
        ctx.beginPath()
        ctx.moveTo(pts[pts.length - 1].x, pts[pts.length - 1].y)
        ctx.lineTo(p.x, p.y)
        ctx.stroke()
        ctx.setLineDash([])
      }
      return
    }
    if (!painting.current || !last.current) return
    const p = pt(e)
    stroke(last.current, p)
    last.current = p
  }

  const onUp = () => {
    painting.current = false
    panning.current = false
    last.current = null
    if (!space.current) setGrabbing(false)
  }

  const onDouble = () => {
    if (toolRef.current === 'polygon') closePoly()
  }

  const clearAll = () => {
    if (!window.confirm('Стереть всю разметку на снимке?')) return
    const cv = canvasRef.current
    cv?.getContext('2d')?.clearRect(0, 0, cv.width, cv.height)
    cancelPoly()
    setDirty(true)
  }

  const zoomBtn = (factor: number) => {
    const st = stageRef.current
    if (st) zoomAt(st.clientWidth / 2, st.clientHeight / 2, factor)
  }

  const save = async () => {
    const cv = canvasRef.current
    if (!cv) return
    setSaving(true)
    setErr(null)
    setMsg(null)
    try {
      await api.saveAnnotation(id, {
        image: cv.toDataURL('image/png'),
        phases,
        width: cv.width,
        height: cv.height,
      })
      setDirty(false)
      setMsg('Разметка сохранена — снимок пойдёт в дообучение.')
    } catch (e) {
      setErr(errorMessage(e))
    } finally {
      setSaving(false)
    }
  }

  const cursor = grabbing ? 'grabbing' : tool === 'erase' ? 'cell' : 'crosshair'

  return (
    <div className={s.page}>
      <div className={s.toolbar}>
        <button className="btn btn-ghost" onClick={guardedBack} type="button">
          ← Очередь
        </button>
        <div className={s.palette}>
          {phases.map((p) => (
            <button
              key={p.key}
              type="button"
              className={`${s.swatch} ${phase === p.key ? s.swatchOn : ''}`}
              aria-pressed={phase === p.key}
              onClick={() => {
                setPhase(p.key)
                if (tool === 'erase') setToolBoth('brush')
              }}
              title={p.label}
            >
              <span className={s.swatchDot} style={{ background: p.color }} />
              {p.label}
            </button>
          ))}
        </div>
        <div className={s.tools}>
          <button className={`${s.toolBtn} ${tool === 'brush' ? s.toolOn : ''}`} aria-pressed={tool === 'brush'} onClick={() => setToolBoth('brush')} type="button" title="Кисть">✎ Кисть</button>
          <button className={`${s.toolBtn} ${tool === 'polygon' ? s.toolOn : ''}`} aria-pressed={tool === 'polygon'} onClick={() => setToolBoth('polygon')} type="button" title="Полигон: клики по вершинам, двойной клик или клик по первой точке — замкнуть">⬡ Полигон</button>
          <button className={`${s.toolBtn} ${tool === 'erase' ? s.toolOn : ''}`} aria-pressed={tool === 'erase'} onClick={() => setToolBoth('erase')} type="button" title="Ластик"><span className={s.eraseDot} /> Ластик</button>
        </div>
        {tool !== 'polygon' && (
          <div className={s.brushCtl}>
            <span className={s.brushLbl}>кисть</span>
            <input type="range" min={4} max={140} value={brush} aria-label="Размер кисти" onChange={(e) => setBrush(Number(e.target.value))} />
            <span className="num" style={{ width: 34 }}>{brush}px</span>
          </div>
        )}
        {tool === 'polygon' && polyLen > 0 && (
          <div className={s.polyCtl}>
            <span className="num">{polyLen} точ.</span>
            <button className="btn" onClick={closePoly} type="button" disabled={polyLen < 3}>Замкнуть</button>
            <button className="btn btn-ghost" onClick={cancelPoly} type="button">Отмена</button>
          </div>
        )}
        <div className={s.zoomCtl}>
          <button className={s.zoomBtn} onClick={() => zoomBtn(1 / 1.25)} type="button" title="Отдалить">−</button>
          <span className={`num ${s.zoomVal}`}>{zoomPct}%</span>
          <button className={s.zoomBtn} onClick={() => zoomBtn(1.25)} type="button" title="Приблизить">+</button>
          <button className={s.zoomBtn} onClick={fit} type="button" title="Вписать в окно">⤢</button>
        </div>
        <div className={s.right}>
          <button className="btn" onClick={clearAll} type="button">Очистить</button>
          <button className="btn btn-primary" onClick={save} type="button" disabled={saving || !dirty}>
            {saving ? 'Сохранение…' : 'Сохранить разметку'}
          </button>
        </div>
      </div>

      {(msg || err) && <div className={err ? s.err : s.msg}>{err || msg}</div>}

      <div
        ref={stageRef}
        className={s.stage}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={onUp}
        onDoubleClick={onDouble}
        onContextMenu={(e) => e.preventDefault()}
        style={{ cursor }}
      >
        <div ref={paneRef} className={s.pane}>
          <img ref={imgRef} className={s.bg} src={urls.preview(id)} onLoad={onImgLoad} alt="снимок для разметки" crossOrigin="anonymous" draggable={false} />
          <canvas ref={canvasRef} className={s.mask} />
          <canvas ref={previewRef} className={s.preview} />
        </div>
      </div>

      <p className={s.hint}>
        <b>Кисть</b> — рисовать; <b>Полигон</b> — клики по вершинам, двойной клик или
        клик по первой точке замыкает и заливает фазой (Escape — отмена);{' '}
        <b>Ластик</b> — стереть. Незакрашенное = «неразмечено» (игнор при обучении).{'  '}
        <b>Тачпад:</b> два пальца — двигать, щипок — масштаб. <b>Мышь:</b> колесо —
        масштаб; двигать — Пробел или средняя кнопка + перетаскивание.
      </p>
    </div>
  )
}
