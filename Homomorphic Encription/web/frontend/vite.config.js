import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// El build va a dist/ y lo sirve FastAPI en la raíz. En desarrollo, el
// dev-server de Vite (5173) redirige /api al backend uvicorn (8000).
//
// Para GitHub Pages el sitio cuelga de un subdirectorio con el nombre del repo
// (https://usuario.github.io/<repo>/), así que hay que compilar con
// VITE_BASE=/<repo>/ para que las rutas de los assets y de los JSON apunten bien.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE || '/',
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1500,
  },
})
