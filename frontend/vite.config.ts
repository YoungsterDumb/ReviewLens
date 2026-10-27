import { devtools } from '@tanstack/devtools-vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// TanStack Devtools must be the first plugin; it strips itself from production builds.
export default defineConfig({
  plugins: [devtools(), tailwindcss(), react()],
  server: { port: 5173 },
})
