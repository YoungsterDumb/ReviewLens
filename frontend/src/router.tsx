import { Link, createRootRoute, createRoute, createRouter } from '@tanstack/react-router'
import { Layout } from './components/Layout'
import { useI18n } from './lib/i18n'
import type { SampleId } from './lib/samples'
import { AnalyzePage } from './routes/AnalyzePage'
import { GeneratePage } from './routes/GeneratePage'
import { InsightsPage } from './routes/InsightsPage'
import { ModelsPage } from './routes/ModelsPage'

function NotFound() {
  const { t } = useI18n()
  return (
    <div className="py-16 text-center">
      <h1 className="m-0 mb-4 text-2xl font-bold">{t('notfound.title')}</h1>
      <Link to="/" className="font-semibold text-ink">
        {t('notfound.back')}
      </Link>
    </div>
  )
}

const rootRoute = createRootRoute({ component: Layout, notFoundComponent: NotFound })

const SAMPLE_IDS: SampleId[] = ['vi_shoe', 'en_laptop', 'vi_phone']

const analyzeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: AnalyzePage,
  validateSearch: (s: Record<string, unknown>): { sample?: SampleId } =>
    SAMPLE_IDS.includes(s.sample as SampleId) ? { sample: s.sample as SampleId } : {},
})

const insightsRoute = createRoute({ getParentRoute: () => rootRoute, path: '/insights', component: InsightsPage })

const generateRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/generate',
  component: GeneratePage,
  validateSearch: (s: Record<string, unknown>): { category?: number } => {
    const n = Number(s.category)
    return Number.isInteger(n) && n >= 0 && n <= 9 ? { category: n } : {}
  },
})

const modelsRoute = createRoute({ getParentRoute: () => rootRoute, path: '/models', component: ModelsPage })

export const routeTree = rootRoute.addChildren([analyzeRoute, insightsRoute, generateRoute, modelsRoute])

export function buildRouter() {
  return createRouter({ routeTree, defaultPreload: 'intent', scrollRestoration: true })
}

declare module '@tanstack/react-router' {
  interface Register {
    router: ReturnType<typeof buildRouter>
  }
}
