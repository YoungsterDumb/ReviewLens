import { useMutation } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { useState } from 'react'
import { api } from '../lib/api'
import { seconds } from '../lib/format'
import { useI18n, type TKey } from '../lib/i18n'
import type { AnalyzeResponse, TraceStep } from '../lib/types'
import { Button, ErrorNote } from './ui'

export function VariantsPanel({ variants, pickedFromText }: { variants: NonNullable<AnalyzeResponse['variants']>; pickedFromText: boolean }) {
  const { t } = useI18n()
  const [images, setImages] = useState(variants.images)
  const regen = useMutation({
    mutationFn: () => api.generate(variants.category_index, variants.probs, images.length, null, false),
    onSuccess: (r) => setImages(r.images),
  })
  const cat = t(`cat.${variants.category_index}` as TKey)
  return (
    <section aria-labelledby="variants-title">
      <h2 id="variants-title" className="m-0 mb-1 text-lg font-bold">
        {t('variants.title')}
      </h2>
      <p className="m-0 mb-3.5 text-[13px] text-ink-2">{pickedFromText ? t('variants.picked', { cat }) : t('variants.lead', { cat })}</p>
      <div className="grid grid-cols-2 gap-2.5">
        {images.map((src, i) => (
          <img key={src.slice(-24) + i} src={src} alt={t('variants.alt', { n: i + 1 })} className="pixelated aspect-square w-full rounded-lg bg-black" />
        ))}
      </div>
      {regen.isError && (
        <div className="mt-3">
          <ErrorNote error={regen.error} />
        </div>
      )}
      <div className="mt-3.5 flex flex-wrap items-center gap-3">
        <Button onClick={() => regen.mutate()} busy={regen.isPending}>
          {t('variants.regenerate')}
        </Button>
        <Link to="/generate" search={{ category: variants.category_index }} className="inline-flex min-h-11 items-center font-semibold text-ink">
          {t('variants.open')}
        </Link>
      </div>
    </section>
  )
}

const STEP_ORDER = ['ingest', 'absa', 'long_context', 'vision', 'fuse', 'generate']

export function TraceGantt({ trace, totalMs }: { trace: TraceStep[]; totalMs: number }) {
  const { t, lang } = useI18n()
  const total = Math.max(totalMs, ...trace.map((s) => s.start_ms + s.ms), 1)
  const steps = [...trace].sort((a, b) => STEP_ORDER.indexOf(a.step) - STEP_ORDER.indexOf(b.step))
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * total)
  return (
    <section aria-labelledby="trace-title">
      <h2 id="trace-title" className="m-0 mb-1 text-lg font-bold">
        {t('trace.title')}
      </h2>
      <p className="m-0 mb-4 text-[13px] text-ink-2">{t('trace.lead', { s: seconds(totalMs, lang) })}</p>
      <ul className="m-0 grid list-none grid-cols-[104px_minmax(0,1fr)] items-center gap-y-3 p-0 text-[13px]">
        {steps.map((s) => {
          const parallel = ['absa', 'long_context', 'vision'].includes(s.step)
          return (
            <li key={s.step} className="contents">
              <span>{t(`step.${s.step}` as TKey)}</span>
              <span className="relative h-3.5 rounded bg-sunken" title={`${seconds(s.ms, lang)} s`}>
                <span
                  className={`absolute inset-y-0 rounded ${parallel ? 'bg-bar' : 'bg-ink'}`}
                  style={{ left: `${(s.start_ms / total) * 100}%`, width: `${Math.max((s.ms / total) * 100, 1.2)}%` }}
                />
                <span className="sr-only">{`${seconds(s.ms, lang)} s`}</span>
              </span>
            </li>
          )
        })}
      </ul>
      <div className="mt-1.5 grid grid-cols-[104px_minmax(0,1fr)] text-xs text-ink-2" aria-hidden>
        <span />
        <span className="flex justify-between">
          {ticks.map((ms) => (
            <span key={ms}>{seconds(ms, lang)} s</span>
          ))}
        </span>
      </div>
    </section>
  )
}
