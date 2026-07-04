import React from 'react'
import ReactDOM from 'react-dom/client'

// Шрифты: интерфейс, дисплейные надписи, моноширинные цифры (кириллица включена).
import '@fontsource/ibm-plex-sans/400.css'
import '@fontsource/ibm-plex-sans/500.css'
import '@fontsource/ibm-plex-sans/600.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import '@fontsource/unbounded/500.css'
import '@fontsource/unbounded/600.css'

import './styles/global.css'
import App from './App'

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            padding: '40px 24px',
            maxWidth: 640,
            margin: '60px auto',
            fontFamily: 'system-ui, sans-serif',
            color: '#26221c',
          }}
        >
          <h1 style={{ fontSize: 20, marginBottom: 12 }}>Что-то пошло не так</h1>
          <p style={{ color: '#5f6c80', marginBottom: 16, lineHeight: 1.6 }}>
            Интерфейс столкнулся с ошибкой отображения. Обновите страницу; если
            повторяется — данные анализа могли прийти в неожиданном виде.
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{ padding: '8px 16px', borderRadius: 8, border: '1px solid #ccc', cursor: 'pointer' }}
          >
            Обновить страницу
          </button>
          <pre style={{ marginTop: 16, fontSize: 11, color: '#aaa', whiteSpace: 'pre-wrap' }}>
            {String(this.state.error?.message ?? this.state.error)}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
)
