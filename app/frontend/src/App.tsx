import { useEffect, useState } from 'react'
import { Header } from './components/Header'
import { ListPage } from './pages/ListPage'
import { AnalysisPage } from './pages/AnalysisPage'
import { FeedbackPage } from './pages/FeedbackPage'
import { AnnotatePage } from './pages/AnnotatePage'
import { useHealth } from './hooks/useHealth'

type Route =
  | { name: 'list' }
  | { name: 'analysis'; id: string }
  | { name: 'feedback' }
  | { name: 'annotate'; id: string }

function parseHash(): Route {
  const h = window.location.hash
  let m = /^#\/a\/([A-Za-z0-9._-]+)/.exec(h)
  if (m) return { name: 'analysis', id: m[1] }
  m = /^#\/annotate\/([A-Za-z0-9._-]+)/.exec(h)
  if (m) return { name: 'annotate', id: m[1] }
  if (/^#\/feedback/.test(h)) return { name: 'feedback' }
  return { name: 'list' }
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash)
  const { health, offline } = useHealth()

  useEffect(() => {
    const onHash = () => setRoute(parseHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const goList = () => {
    window.location.hash = '/'
  }
  const goAnalysis = (id: string) => {
    window.location.hash = `/a/${id}`
  }
  const goFeedback = () => {
    window.location.hash = '/feedback'
  }
  const goAnnotate = (id: string) => {
    window.location.hash = `/annotate/${id}`
  }

  return (
    <>
      <Header
        health={health}
        offline={offline}
        onHome={goList}
        onFeedback={goFeedback}
        active={route.name}
      />
      {route.name === 'list' && <ListPage onOpen={goAnalysis} />}
      {route.name === 'analysis' && (
        <AnalysisPage key={route.id} id={route.id} onBack={goList} onAnnotate={goAnnotate} />
      )}
      {route.name === 'feedback' && (
        <FeedbackPage onOpen={goAnalysis} onAnnotate={goAnnotate} onBack={goList} />
      )}
      {route.name === 'annotate' && (
        <AnnotatePage key={route.id} id={route.id} onBack={goFeedback} />
      )}
    </>
  )
}
