import { TanStackDevtools } from '@tanstack/react-devtools'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtoolsPanel } from '@tanstack/react-query-devtools'
import { RouterProvider } from '@tanstack/react-router'
import { TanStackRouterDevtoolsPanel } from '@tanstack/react-router-devtools'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { I18nProvider } from './lib/i18n'
import { buildRouter } from './router'
import './styles.css'

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 }, mutations: { retry: 0 } },
})
const router = buildRouter()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <RouterProvider router={router} />
      </I18nProvider>
      {/* Development only: the TanStack Devtools Vite plugin removes this from production builds. */}
      <TanStackDevtools
        plugins={[
          { name: 'TanStack Query', render: <ReactQueryDevtoolsPanel /> },
          { name: 'TanStack Router', render: <TanStackRouterDevtoolsPanel router={router} /> },
        ]}
      />
    </QueryClientProvider>
  </StrictMode>,
)
