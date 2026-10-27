import type {
  AnalyzeResponse,
  BatchResponse,
  ExplainImageResponse,
  ExplainTextResponse,
  GenerateResponse,
  HealthResponse,
  MetricsResponse,
  Probs,
} from './types'

export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/+$/, '')

export class ApiError extends Error {
  readonly status: number
  readonly code: string

  constructor(status: number, code: string, message: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 180_000): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal })
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === 'AbortError'
    throw new ApiError(0, aborted ? 'timeout' : 'network', aborted ? 'The server took too long to answer.' : 'Cannot reach the server.')
  } finally {
    clearTimeout(timer)
  }
  if (!res.ok) {
    let code = 'http_error'
    let message = `Request failed (${res.status}).`
    try {
      const body = await res.json()
      code = body?.error?.code ?? code
      message = body?.error?.message ?? message
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, code, message)
  }
  return (await res.json()) as T
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export interface AnalyzeInput {
  text?: string
  lang?: 'auto' | 'vi' | 'en'
  image?: File | Blob | null
  audio?: File | Blob | null
  category?: number | null
  seed?: number | null
}

export function analyzeForm(input: AnalyzeInput): FormData {
  const fd = new FormData()
  if (input.text?.trim()) fd.append('text', input.text)
  fd.append('lang', input.lang ?? 'auto')
  if (input.image) fd.append('image', input.image, input.image instanceof File ? input.image.name : 'photo.png')
  if (input.audio) fd.append('audio', input.audio, input.audio instanceof File ? input.audio.name : 'voice-note.webm')
  if (input.category != null) fd.append('category', String(input.category))
  if (input.seed != null) fd.append('seed', String(input.seed))
  return fd
}

export const api = {
  health: () => request<HealthResponse>('/health', {}, 90_000),
  warmup: () => request<HealthResponse>('/warmup', { method: 'POST' }, 90_000),
  metrics: () => request<MetricsResponse>('/metrics'),
  analyze: (input: AnalyzeInput) => request<AnalyzeResponse>('/analyze', { method: 'POST', body: analyzeForm(input) }),
  explainText: (text: string, aspect: string) => request<ExplainTextResponse>('/explain/text', json({ text, aspect })),
  explainImage: (image: Blob) => {
    const fd = new FormData()
    fd.append('image', image, 'photo.png')
    return request<ExplainImageResponse>('/explain/image', { method: 'POST', body: fd })
  },
  vqa: (image: Blob, question: string) => {
    const fd = new FormData()
    fd.append('image', image, 'photo.png')
    fd.append('question', question)
    return request<{ question: string; answer: string; confidence: number | null }>('/vqa', { method: 'POST', body: fd })
  },
  generate: (category: number, sentiment: Probs, n: number, seed: number | null, sweep = true) =>
    request<GenerateResponse>('/generate', json({ category, sentiment, n, seed, sweep })),
  batch: (texts: string[]) => request<BatchResponse>('/absa/batch', json({ texts }), 300_000),
}
