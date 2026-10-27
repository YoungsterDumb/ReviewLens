import { useMutation } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { Check, Circle, CircleAlert, LoaderCircle, Mic } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { AspectList } from '../components/AspectList'
import { ImagePicker, VoiceRecorder, type VoiceNote } from '../components/Inputs'
import { PhotoPanel } from '../components/PhotoPanel'
import { TraceGantt, VariantsPanel } from '../components/ResultPanels'
import { ReviewText } from '../components/ReviewText'
import { SentimentBar } from '../components/SentimentBar'
import { Button, ErrorNote, SectionTitle } from '../components/ui'
import { api } from '../lib/api'
import { clock, overallLabel, pct, priorityReasons } from '../lib/format'
import { useServerHealth } from '../lib/health'
import { useI18n, type TKey } from '../lib/i18n'
import { SAMPLES, type SampleId } from '../lib/samples'
import type { AnalyzeResponse, HealthResponse, ModelState } from '../lib/types'

const MAX_CHARS = 20_000

interface Draft {
  text: string
  image: File | null
  voice: VoiceNote | null
  lang: 'auto' | 'vi' | 'en'
}

const EMPTY: Draft = { text: '', image: null, voice: null, lang: 'auto' }

export function AnalyzePage() {
  const search = useSearch({ from: '/' })
  const navigate = useNavigate({ from: '/' })
  const [draft, setDraft] = useState<Draft>(EMPTY)
  const [result, setResult] = useState<{ data: AnalyzeResponse; draft: Draft } | null>(null)
  const analyze = useMutation({
    mutationFn: (d: Draft) => api.analyze({ text: d.text, lang: d.lang, image: d.image, audio: d.voice?.blob ?? null }),
    onSuccess: (data, d) => {
      setResult({ data, draft: d })
      window.scrollTo({ top: 0, behavior: 'smooth' })
    },
  })

  const loadSample = async (id: SampleId) => {
    const sample = SAMPLES.find((s) => s.id === id)
    if (!sample) return
    let image: File | null = null
    if (sample.image) {
      try {
        const blob = await (await fetch(sample.image)).blob()
        image = new File([blob], sample.image.split('/').pop() ?? 'sample.png', { type: blob.type || 'image/png' })
      } catch {
        image = null
      }
    }
    setResult(null)
    setDraft({ text: sample.text, image, voice: null, lang: 'auto' })
  }

  const loaded = useRef<string | null>(null)
  useEffect(() => {
    if (search.sample && loaded.current !== search.sample) {
      loaded.current = search.sample
      void loadSample(search.sample)
      void navigate({ search: {}, replace: true })
    }
  }, [search.sample, navigate])

  if (result) {
    return (
      <Results
        data={result.data}
        draft={result.draft}
        onEdit={() => {
          setDraft(result.draft)
          setResult(null)
        }}
        onNew={() => {
          setDraft(EMPTY)
          setResult(null)
          analyze.reset()
        }}
      />
    )
  }
  return <Compose draft={draft} setDraft={setDraft} onSubmit={(d) => analyze.mutate(d)} pending={analyze.isPending} error={analyze.error} onSample={loadSample} />
}

function Compose({ draft, setDraft, onSubmit, pending, error, onSample }: {
  draft: Draft
  setDraft: (d: Draft) => void
  onSubmit: (d: Draft) => void
  pending: boolean
  error: unknown
  onSample: (id: SampleId) => void
}) {
  const { t, lang } = useI18n()
  const { health, status } = useServerHealth()
  const empty = !draft.text.trim() && !draft.image && !draft.voice
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!empty && !pending) onSubmit(draft)
  }
  return (
    <>
      <div className="mb-7 max-w-[720px]">
        <h1 className="m-0 text-[30px] font-bold leading-tight tracking-[-0.02em] md:text-4xl">{t('analyze.title')}</h1>
        <p className="m-0 mt-2.5 text-[17px] text-ink-2">{t('analyze.lead')}</p>
      </div>
      <div className="flex flex-wrap items-start gap-8">
        <form onSubmit={submit} className="min-w-0 flex-[7_1_560px] rounded-2xl border border-line bg-surface p-5 md:px-8 md:py-7" aria-busy={pending}>
          <div className="mb-2 flex items-baseline justify-between">
            <label htmlFor="review" className="font-semibold">
              {t('analyze.review')}
            </label>
            <span className="text-[13px] text-ink-2">
              {t('analyze.chars', {
                n: new Intl.NumberFormat(lang === 'vi' ? 'vi-VN' : 'en-US').format(draft.text.length),
                max: new Intl.NumberFormat(lang === 'vi' ? 'vi-VN' : 'en-US').format(MAX_CHARS),
              })}
            </span>
          </div>
          <textarea
            id="review"
            rows={7}
            maxLength={MAX_CHARS}
            value={draft.text}
            onChange={(e) => setDraft({ ...draft, text: e.target.value })}
            placeholder={t('analyze.placeholder')}
            className="block w-full resize-y rounded-xl border border-line-strong bg-paper px-4 py-4 font-serif text-[19px] leading-[1.7] text-ink placeholder:text-ink-3"
          />
          <div className="mt-4 flex flex-wrap gap-4">
            <ImagePicker file={draft.image} onChange={(image) => setDraft({ ...draft, image })} />
            <VoiceRecorder value={draft.voice} onChange={(voice) => setDraft({ ...draft, voice })} />
          </div>
          <div className="mt-5 flex flex-wrap items-end gap-3">
            <div className="flex-[0_1_220px]">
              <label htmlFor="lang" className="mb-1.5 block text-[13px] font-semibold">
                {t('analyze.lang')}
              </label>
              <select
                id="lang"
                value={draft.lang}
                onChange={(e) => setDraft({ ...draft, lang: e.target.value as Draft['lang'] })}
                className="min-h-11 w-full rounded-[10px] border border-line-strong bg-surface px-3 text-ink"
              >
                <option value="auto">{t('analyze.langAuto')}</option>
                <option value="vi">Tiếng Việt</option>
                <option value="en">English</option>
              </select>
            </div>
            <div className="flex-1" />
            <Button type="submit" variant="primary" size="lg" busy={pending} disabled={empty || pending}>
              {pending ? t('analyze.submitting') : t('analyze.submit')}
            </Button>
          </div>
          {status !== 'ready' && <p className="m-0 mt-2.5 text-right text-[13px] text-ink-2">{t('analyze.coldHint')}</p>}
          {error != null && (
            <div className="mt-4">
              <ErrorNote error={error} onRetry={() => onSubmit(draft)} />
            </div>
          )}
          <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-sunken pt-4">
            <span className="mr-1 text-[13px] text-ink-2">{t('analyze.samples')}</span>
            {SAMPLES.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => onSample(s.id)}
                className="min-h-9 cursor-pointer rounded-full border border-line-strong bg-surface px-3.5 text-[13px] text-ink hover:bg-sunken"
              >
                {t(`sample.${s.id}` as TKey)}
              </button>
            ))}
          </div>
        </form>
        <aside className="flex min-w-0 flex-[4_1_320px] flex-col gap-7">
          <section>
            <h2 className="m-0 mb-3.5 text-lg font-bold">{t('steps.title')}</h2>
            <ol className="m-0 flex list-none flex-col gap-3.5 p-0">
              {(['read', 'score', 'explain', 'generate'] as const).map((k, i) => (
                <li key={k} className="flex gap-3">
                  <span aria-hidden className="inline-flex size-[26px] flex-none items-center justify-center rounded-full border-[1.5px] border-ink text-[13px] font-bold">
                    {i + 1}
                  </span>
                  <span>
                    <b className="font-semibold">{t(`steps.${k}` as TKey)}</b>
                    <br />
                    <span className="text-ink-2">{t(`steps.${k}Body` as TKey)}</span>
                  </span>
                </li>
              ))}
            </ol>
          </section>
          {health && status !== 'ready' && <WarmupCard health={health} />}
        </aside>
      </div>
    </>
  )
}

const MODEL_ROWS: { key: TKey; states: (h: HealthResponse) => ModelState }[] = [
  { key: 'model.absa', states: (h) => (h.models.ate?.state === 'ready' && h.models.atsc?.state === 'ready' ? 'ready' : worst([h.models.ate?.state, h.models.atsc?.state])) },
  { key: 'model.longctx', states: (h) => h.models.longctx?.state ?? 'idle' },
  { key: 'model.asr', states: (h) => h.models.asr?.state ?? 'idle' },
  { key: 'model.vision', states: (h) => h.models.vision?.state ?? 'idle' },
  { key: 'model.generator', states: (h) => h.models.generator?.state ?? 'idle' },
]

function worst(states: (ModelState | undefined)[]): ModelState {
  if (states.includes('error')) return 'error'
  if (states.includes('loading')) return 'loading'
  return 'idle'
}

function WarmupCard({ health }: { health: HealthResponse }) {
  const { t } = useI18n()
  const icon = { ready: Check, loading: LoaderCircle, idle: Circle, error: CircleAlert }
  const tone = { ready: 'text-pos-ink', loading: 'text-neu-ink', idle: 'text-ink-2', error: 'text-neg-ink' }
  return (
    <section className="rounded-2xl border border-line bg-surface p-5" aria-live="polite">
      <h2 className="m-0 mb-1 text-[15px] font-bold">{t('warm.title')}</h2>
      <p className="m-0 mb-3.5 text-[13px] text-ink-2">{t('warm.body')}</p>
      <ul className="m-0 flex list-none flex-col gap-2.5 p-0 text-sm">
        {MODEL_ROWS.map((row) => {
          const s = row.states(health)
          const Icon = icon[s]
          return (
            <li key={row.key} className="flex items-center gap-2.5">
              <Icon aria-hidden className={`size-[18px] ${tone[s]} ${s === 'loading' ? 'animate-spin' : ''}`} />
              <span className="flex-1">{t(row.key)}</span>
              <span className={`text-[13px] font-semibold ${tone[s]}`}>{t(`state.${s}` as TKey)}</span>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

function Results({ data, draft, onEdit, onNew }: { data: AnalyzeResponse; draft: Draft; onEdit: () => void; onNew: () => void }) {
  const { t, lang } = useI18n()
  const [active, setActive] = useState<number | null>(null)
  const sub =
    (data.language === 'vi' ? t('results.subVi') : t('results.subEn')) +
    (draft.image ? t('results.subPhoto') : '') +
    (data.audio_seconds ? t('results.subVoice', { time: clock(data.audio_seconds) }) : '') +
    '.'
  const tox = data.toxicity
  const toxic = tox ? tox.max_score >= 0.5 : false
  const priorityTone = { high: 'bg-neg-mark text-neg-ink', medium: 'bg-neu-mark text-neu-ink', low: 'bg-pos-mark text-pos-ink' }[data.priority.level]
  return (
    <>
      <div className="mb-6 flex flex-wrap items-end gap-4">
        <div className="min-w-0 flex-[1_1_420px]">
          <h1 className="m-0 text-[28px] font-bold leading-tight tracking-[-0.015em]">{t('results.title')}</h1>
          <p className="m-0 mt-1.5 text-ink-2">{sub}</p>
        </div>
        <div className="flex flex-wrap gap-2.5">
          <Button onClick={onEdit}>{t('results.edit')}</Button>
          <Button variant="primary" onClick={onNew}>
            {t('results.new')}
          </Button>
        </div>
      </div>
      {data.warnings.map((w) => (
        <p key={w} className="m-0 mb-3 rounded-xl bg-neu-soft px-4 py-2.5 text-[13px] text-neu-ink">
          {t(`warning.${w.split(':')[0]}` as TKey)}
        </p>
      ))}

      <div className="flex flex-wrap items-start gap-8">
        <article className="min-w-0 flex-[7_1_560px] rounded-2xl border border-line bg-surface px-5 pb-6 pt-6 md:px-9 md:pt-8">
          <div className="mb-4 flex flex-wrap items-center gap-2 text-[13px] text-ink-2">
            <span className="rounded-full bg-sunken px-2.5 py-0.5 font-medium text-ink">{data.language === 'vi' ? 'Tiếng Việt' : 'English'}</span>
            <span className="rounded-full bg-sunken px-2.5 py-0.5">{t('results.words', { n: data.words })}</span>
            {data.long_context && <span className="rounded-full bg-sunken px-2.5 py-0.5">{t('results.long', { n: data.tokens })}</span>}
            {data.transcript && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-sunken px-2.5 py-0.5">
                <Mic aria-hidden className="size-3.5" />
                {t('results.transcribed')}
              </span>
            )}
          </div>
          {data.text ? <ReviewText text={data.text} aspects={data.aspects} active={active} /> : <p className="m-0 text-ink-2">{t('results.noText')}</p>}
          <div className="my-6 h-px bg-line" />
          <div className="flex flex-wrap gap-7">
            <section className="min-w-0 flex-[2_1_260px]">
              <h2 className="m-0 mb-1.5 text-[13px] font-medium text-ink-2">{t('overall.title')}</h2>
              <p className="m-0 mb-2.5 text-xl font-bold">{overallLabel(data.overall, t)}</p>
              <SentimentBar probs={data.overall.probs} />
            </section>
            <section className="min-w-0 flex-[1_1_180px]">
              <h2 className="m-0 mb-1.5 text-[13px] font-medium text-ink-2">{t('priority.title')}</h2>
              <p className="m-0 mb-1.5">
                <span className={`inline-flex rounded-full px-3 py-1 text-[15px] font-bold ${priorityTone}`}>{t(`priority.${data.priority.level}`)}</span>
              </p>
              <p className="m-0 text-[13px] text-ink-2">{priorityReasons(data.priority, t)}</p>
            </section>
            <section className="min-w-0 flex-[1_1_140px]">
              <h2 className="m-0 mb-1.5 text-[13px] font-medium text-ink-2">{t('toxicity.title')}</h2>
              {tox ? (
                <>
                  <p className={`m-0 mb-1.5 text-xl font-bold ${toxic ? 'text-neg-ink' : ''}`}>{toxic ? t('toxicity.yes') : t('toxicity.no')}</p>
                  <p className="m-0 text-[13px] text-ink-2">{t('toxicity.max', { p: pct(tox.max_score, lang), label: t(`tox.${tox.max_label}` as TKey) })}</p>
                </>
              ) : (
                <p className="m-0 text-[13px] text-ink-2">{t('toxicity.na')}</p>
              )}
            </section>
          </div>
        </article>
        <aside className="min-w-0 flex-[5_1_360px]" aria-label={t('aspects.title', { n: data.aspects.length })}>
          <SectionTitle aside={data.aspects.length ? t('aspects.order') : undefined}>{t('aspects.title', { n: data.aspects.length })}</SectionTitle>
          <AspectList aspects={data.aspects} active={active} onActive={setActive} />
        </aside>
      </div>

      <div className="mt-11 grid gap-x-7 gap-y-8 border-t border-line-strong pt-6 md:grid-cols-2 xl:grid-cols-3">
        {draft.image && data.image ? (
          <PhotoPanel image={draft.image} result={data.image} />
        ) : (
          <section>
            <h2 className="m-0 mb-2 text-lg font-bold">{t('photo.title')}</h2>
            <p className="m-0 text-sm text-ink-2">{t('photo.none')}</p>
          </section>
        )}
        {data.variants && <VariantsPanel variants={data.variants} pickedFromText={!data.image || data.image.category_index == null} />}
        <TraceGantt trace={data.trace} totalMs={data.total_ms} />
      </div>
    </>
  )
}
