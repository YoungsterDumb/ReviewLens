import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider, createMemoryHistory, createRouter } from '@tanstack/react-router'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ReviewText } from './components/ReviewText'
import { I18nProvider } from './lib/i18n'
import type { AnalyzeResponse, Aspect, HealthResponse } from './lib/types'
import { routeTree } from './router'

const HEALTH: HealthResponse = {
  status: 'ok',
  mode: 'live',
  version: '1.0.0',
  models: Object.fromEntries(['ate', 'atsc', 'longctx', 'asr', 'vision', 'generator'].map((m) => [m, { state: 'ready', error: null }])),
}

const aspect = (term: string, start: number, sentiment: Aspect['sentiment']): Aspect => ({
  term, start, end: start + term.length, category: null, sentence: 'The screen is great. The battery is bad.',
  sentiment, confidence: 0.9, probs: { negative: 0.05, neutral: 0.05, positive: 0.9 }, extraction_score: 0.95,
})

const ANALYZE: AnalyzeResponse = {
  language: 'en',
  text: 'The screen is great. The battery is bad.',
  transcript: null,
  audio_seconds: null,
  tokens: 12,
  words: 8,
  long_context: false,
  aspects: [aspect('screen', 4, 'positive'), aspect('battery', 25, 'negative')],
  overall: { label: 'positive', mixed: true, probs: { positive: 0.5, neutral: 0.1, negative: 0.4 } },
  toxicity: { scores: { toxic: 0.02 }, max_label: 'toxic', max_score: 0.02, tokens: 12, coverage: 1 },
  image: null,
  priority: { level: 'medium', reasons: ['negative_aspects:1'] },
  variants: { category: 'Bag', category_index: 8, probs: { positive: 0.5, neutral: 0.1, negative: 0.4 }, images: ['data:image/png;base64,AAAA'] },
  trace: [
    { step: 'ingest', start_ms: 0, ms: 5 },
    { step: 'absa', start_ms: 5, ms: 40 },
    { step: 'generate', start_ms: 50, ms: 10 },
  ],
  total_ms: 60,
  warnings: [],
}

function mockApi(overrides: Record<string, unknown> = {}) {
  const calls: string[] = []
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input)
    const path = new URL(url).pathname
    calls.push(path)
    const body =
      overrides[path] ??
      {
        '/health': HEALTH,
        '/warmup': HEALTH,
        '/analyze': ANALYZE,
        '/explain/text': {
          aspect: 'battery', label: 'negative', probs: { negative: 0.9, neutral: 0.05, positive: 0.05 },
          weights: [{ word: 'bad', weight: 0.41 }, { word: 'great', weight: -0.05 }], fidelity: 0.95, r2: 0.8, num_samples: 300,
        },
      }[path]
    if (!body) return new Response(JSON.stringify({ error: { code: 'not_found', message: 'nope' } }), { status: 404 })
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
  })
  return calls
}

function renderApp(path = '/', lang: 'vi' | 'en' = 'en') {
  const router = createRouter({ routeTree, history: createMemoryHistory({ initialEntries: [path] }) })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <I18nProvider lang={lang}>
        <RouterProvider router={router} />
      </I18nProvider>
    </QueryClientProvider>,
  )
  return router
}

describe('ReviewText', () => {
  it('marks each aspect with its sentiment and a footnote number', () => {
    const { container } = render(<ReviewText text={ANALYZE.text} aspects={ANALYZE.aspects} active={1} />)
    const marks = container.querySelectorAll('mark')
    expect([...marks].map((m) => m.textContent)).toEqual(['screen', 'battery'])
    expect(marks[0].className).toContain('mark-positive')
    expect(marks[1].className).toContain('mark-negative')
    expect(marks[1].className).toContain('mark-active')
    expect(container.querySelectorAll('sup')[1].textContent).toBe('2')
    expect(container.textContent).toBe(ANALYZE.text.replace('screen', 'screen1').replace('battery', 'battery2'))
  })
})

describe('app', () => {
  beforeEach(() => window.scrollTo = vi.fn() as never)
  afterEach(() => vi.restoreAllMocks())

  it('analyzes a review end to end and explains an aspect', async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderApp('/')
    expect(await screen.findByRole('heading', { name: 'Read every review closely' })).toBeInTheDocument()
    expect(await screen.findByText('Models ready')).toBeInTheDocument()

    const submit = screen.getByRole('button', { name: 'Analyze' })
    expect(submit).toBeDisabled()
    await user.type(screen.getByLabelText('Review'), 'The screen is great. The battery is bad.')
    await user.click(submit)

    expect(await screen.findByRole('heading', { name: 'Analysis results' })).toBeInTheDocument()
    expect(screen.getByText('Positive, with complaints')).toBeInTheDocument()
    expect(screen.getByText('Medium')).toBeInTheDocument()
    const list = screen.getByRole('complementary', { name: '2 aspects' })
    const items = within(list).getAllByRole('listitem')
    expect(items).toHaveLength(2)
    expect(calls).toContain('/warmup')

    await user.click(within(items[1]).getByRole('button', { name: /Why\?/ }))
    expect(await screen.findByText('Words behind “Negative”')).toBeInTheDocument()
    expect(screen.getByText('Fidelity 95%')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Analyze a new review' }))
    expect(await screen.findByRole('heading', { name: 'Read every review closely' })).toBeInTheDocument()
  })

  it('shows the server error message and keeps the draft', async () => {
    mockApi({ '/analyze': undefined })
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const path = new URL(String(input)).pathname
      if (path === '/analyze') {
        return new Response(JSON.stringify({ error: { code: 'model_unavailable', message: 'x' } }), { status: 503 })
      }
      return new Response(JSON.stringify(HEALTH), { status: 200 })
    })
    const user = userEvent.setup()
    renderApp('/')
    await user.type(await screen.findByLabelText('Review'), 'Battery is bad')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('A model is not ready yet.')
    expect(screen.getByLabelText('Review')).toHaveValue('Battery is bad')
  })

  it('renders every page in Vietnamese with navigation', async () => {
    mockApi({
      '/metrics': {
        updated_at: null, source: 'local', models: { ate: 'DatDuc/reviewlens-ate' },
        metrics: [{ key: 'vae_recon_mse', value: 0.0087, target: 0.01, direction: 'lower', met: true, details: {} }],
      },
    })
    const user = userEvent.setup()
    renderApp('/', 'vi')
    expect(await screen.findByRole('heading', { name: 'Đọc kỹ từng đánh giá' })).toBeInTheDocument()
    const nav = screen.getAllByRole('navigation', { name: 'Chính' })[0]
    await user.click(within(nav).getByRole('link', { name: 'Tổng hợp' }))
    expect(await screen.findByRole('heading', { name: 'Tổng hợp khía cạnh' })).toBeInTheDocument()
    await user.click(within(nav).getByRole('link', { name: 'Tạo ảnh' }))
    expect(await screen.findByRole('heading', { name: 'Tạo biến thể ảnh' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Giày thể thao' })).toHaveAttribute('aria-pressed', 'true')
    await user.click(within(nav).getByRole('link', { name: 'Mô hình' }))
    expect(await screen.findByText('0,0087')).toBeInTheDocument()
    expect(screen.getByText('Đạt')).toBeInTheDocument()
  })

  it('opens Generate with the category from the link', async () => {
    mockApi()
    renderApp('/generate?category=3')
    expect(await screen.findByRole('button', { name: 'Dress' })).toHaveAttribute('aria-pressed', 'true')
  })
})
