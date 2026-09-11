import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The React plugin gives us fast refresh; the Tailwind plugin compiles the utility classes.
// Everything under /api is forwarded to the FastAPI backend so the browser sees a single
// origin during development (the CORS middleware in the backend covers the direct case).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
