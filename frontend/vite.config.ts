import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'
import path from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/training': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
    headers: {
      'Content-Security-Policy': 'default-src \'self\'; script-src \'self\' \'unsafe-eval\' \'unsafe-inline\' http://localhost:3000; style-src \'self\' \'unsafe-inline\' https://fonts.googleapis.com; font-src \'self\' data: https://fonts.gstatic.com; img-src \'self\' data: blob:; connect-src \'self\' http://localhost:8000 ws://localhost:3000;',
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['@headlessui/react', '@heroicons/react'],
          charts: ['chart.js', 'react-chartjs-2'],
          three: ['three', '@react-three/fiber', '@react-three/drei'],
        },
      },
    },
  },
})