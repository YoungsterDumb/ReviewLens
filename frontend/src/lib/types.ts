export type Sentiment = 'negative' | 'neutral' | 'positive'
export type Probs = Record<Sentiment, number>
export type Lang = 'vi' | 'en'

export interface Aspect {
  term: string
  start: number
  end: number
  category: string | null
  sentence: string
  sentiment: Sentiment
  confidence: number
  probs: Probs
  extraction_score: number
}

export interface Overall {
  label: Sentiment
  mixed: boolean
  probs: Probs
}

export interface Toxicity {
  scores: Record<string, number>
  max_label: string
  max_score: number
  tokens: number
  coverage: number
}

export interface QA {
  question: string
  answer: string
  confidence?: number | null
}

export interface ImageResult {
  category: string | null
  category_index: number | null
  category_confidence: number
  condition: { defective: number; good: number }
  answers: QA[]
  aspect_alignment: { term: string; score: number }[]
}

export interface Priority {
  level: 'low' | 'medium' | 'high'
  reasons: string[]
}

export interface TraceStep {
  step: string
  start_ms: number
  ms: number
}

export interface AnalyzeResponse {
  language: Lang
  text: string
  transcript: string | null
  audio_seconds: number | null
  tokens: number
  words: number
  long_context: boolean
  aspects: Aspect[]
  overall: Overall
  toxicity: Toxicity | null
  image: ImageResult | null
  priority: Priority
  variants: { category: string; category_index: number; probs: Probs; images: string[] } | null
  trace: TraceStep[]
  total_ms: number
  warnings: string[]
}

export interface ExplainTextResponse {
  aspect: string
  label: Sentiment
  probs: Probs
  weights: { word: string; weight: number }[]
  fidelity: number
  r2: number
  num_samples: number
}

export interface ExplainImageResponse {
  label: 'defective' | 'good'
  probability: number
  heatmap: string
  fidelity: number
  r2: number
  num_samples: number
}

export interface GenerateResponse {
  category: string
  category_index: number
  probs: Probs
  seed: number
  images: string[]
  sweep: string[]
}

export interface BatchReview {
  index: number
  language: Lang
  text: string
  aspects: Aspect[]
  overall: Overall
  priority: Priority
}

export interface SummaryRow {
  key: string
  label: string
  is_category: boolean
  mentions: number
  positive: number
  neutral: number
  negative: number
  net: number
  example: string
}

export interface BatchResponse {
  reviews: BatchReview[]
  summary: SummaryRow[]
  total_ms: number
}

export interface Metric {
  key: string
  value: number | null
  target: number
  direction: 'higher' | 'lower'
  met: boolean | null
  details: Record<string, number | string | null>
}

export interface MetricsResponse {
  updated_at: string | null
  source: string
  metrics: Metric[]
  models: Record<string, string>
}

export type ModelState = 'idle' | 'loading' | 'ready' | 'error'

export interface HealthResponse {
  status: 'ok'
  mode: 'live' | 'stub'
  version: string
  models: Record<string, { state: ModelState; error: string | null }>
}
