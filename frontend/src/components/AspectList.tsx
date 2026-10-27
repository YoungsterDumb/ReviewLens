import { useQuery } from '@tanstack/react-query'
import { ChevronDown } from 'lucide-react'
import { useId, useState } from 'react'
import { api } from '../lib/api'
import { aspectCategoryLabel, pct, signed } from '../lib/format'
import { useI18n } from '../lib/i18n'
import type { Aspect, ExplainTextResponse } from '../lib/types'
import { ErrorNote, NumberBadge, SentimentLabel, Spinner } from './ui'

export function WordWeights({ data }: { data: ExplainTextResponse }) {
  const { t, lang } = useI18n()
  const label = t(`label.${data.label}`)
  const max = Math.max(...data.weights.map((w) => Math.abs(w.weight)), 1e-6)
  return (
    <div>
      <p className="m-0 mb-1 font-semibold">{t('explain.title', { label })}</p>
      <p className="m-0 mb-3 text-[13px] text-ink-2">{t('explain.lead')}</p>
      <div className="grid grid-cols-[minmax(72px,112px)_minmax(0,1fr)_minmax(0,1fr)_52px] items-center gap-y-2 text-[13px]">
        {data.weights.map((w) => {
          const width = `${(Math.abs(w.weight) / max) * 100}%`
          const toward = w.weight > 0
          return (
            <div key={w.word} className="contents">
              <span className="truncate font-serif text-[15px]" title={w.word}>
                {w.word}
              </span>
              <span className="flex justify-end">
                {!toward && <span className="h-3 rounded-l bg-against" style={{ width }} />}
              </span>
              <span className="flex border-l border-line-strong">
                {toward && <span className="h-3 rounded-r bg-neg" style={{ width }} />}
              </span>
              <span className={`text-right font-semibold ${toward ? 'text-neg-ink' : 'text-ink-2'}`}>{signed(w.weight, lang)}</span>
            </div>
          )
        })}
      </div>
      <div className="mt-3 flex flex-wrap gap-4 text-xs text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-sm bg-neg" />
          {t('explain.toward', { label })}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-sm bg-against" />
          {t('explain.against')}
        </span>
      </div>
      <div className="mt-3.5 flex flex-wrap gap-2.5 text-[13px]">
        <span className={`rounded-lg px-2.5 py-1.5 font-semibold ${data.fidelity >= 0.9 ? 'bg-pos-soft text-pos-ink' : 'bg-neu-soft text-neu-ink'}`}>
          {t('explain.fidelity', { p: pct(data.fidelity, lang) })}
        </span>
        <span className="rounded-lg bg-sunken px-2.5 py-1.5 text-ink-2">{t('explain.r2', { v: signed(data.r2, lang).replace('+', '') })}</span>
        <span className="rounded-lg bg-sunken px-2.5 py-1.5 text-ink-2">{t('explain.samples', { n: data.num_samples })}</span>
      </div>
    </div>
  )
}

function AspectItem({ aspect, index, active, onActive }: { aspect: Aspect; index: number; active: boolean; onActive: (i: number | null) => void }) {
  const { t, lang } = useI18n()
  const [open, setOpen] = useState(false)
  const panelId = useId()
  const explain = useQuery({
    queryKey: ['explain-text', aspect.sentence, aspect.term],
    queryFn: () => api.explainText(aspect.sentence, aspect.term),
    enabled: open,
    staleTime: Infinity,
    retry: 1,
  })
  const border = open ? (aspect.sentiment === 'negative' ? 'border-2 border-neg' : aspect.sentiment === 'positive' ? 'border-2 border-pos' : 'border-2 border-neu') : 'border border-line'
  return (
    <li
      className={`rounded-xl bg-surface px-4 py-3.5 ${border} ${active && !open ? 'ring-2 ring-line-strong' : ''}`}
      onMouseEnter={() => onActive(index)}
      onMouseLeave={() => onActive(null)}
      onFocus={() => onActive(index)}
      onBlur={() => onActive(null)}
    >
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5">
        <NumberBadge n={index + 1} sentiment={aspect.sentiment} />
        <span className="min-w-0 flex-1 font-semibold">
          {aspect.term}
          {aspect.category && <span className="ml-2 text-xs font-medium text-ink-2">{aspectCategoryLabel(aspect.category, lang)}</span>}
        </span>
        <SentimentLabel sentiment={aspect.sentiment} confidence={aspect.confidence} />
        <button
          type="button"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((o) => !o)}
          className={`inline-flex min-h-8 cursor-pointer items-center gap-1 rounded-lg px-2.5 text-[13px] ${open ? 'bg-primary text-on-primary' : 'border border-line-strong bg-surface text-ink hover:bg-sunken'}`}
        >
          {open ? t('aspects.hide') : t('aspects.why')}
          <ChevronDown aria-hidden className={`size-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>
      {open && (
        <div id={panelId} className="mt-3.5 border-t border-sunken pt-3.5">
          {explain.isPending && <Spinner label={t('explain.loading')} />}
          {explain.isError && <ErrorNote error={explain.error} onRetry={() => explain.refetch()} />}
          {explain.data && <WordWeights data={explain.data} />}
        </div>
      )}
    </li>
  )
}

export function AspectList({ aspects, onActive, active }: { aspects: Aspect[]; active: number | null; onActive: (i: number | null) => void }) {
  const { t } = useI18n()
  if (!aspects.length) {
    return <p className="m-0 rounded-xl border border-dashed border-line-strong bg-surface p-4 text-sm text-ink-2">{t('aspects.empty')}</p>
  }
  return (
    <ol className="m-0 flex list-none flex-col gap-1.5 p-0">
      {aspects.map((a, i) => (
        <AspectItem key={`${a.start}-${a.end}`} aspect={a} index={i} active={active === i} onActive={onActive} />
      ))}
    </ol>
  )
}
