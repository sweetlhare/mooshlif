import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Бэкенд (FastAPI) в dev-режиме; SSE проксируется корректно.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'es2020',
    rollupOptions: {
      output: {
        manualChunks: {
          osd: ['openseadragon'],
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
})
