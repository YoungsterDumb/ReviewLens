import { useMutation } from '@tanstack/react-query'
import { Download, Upload } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import { CountBar } from '../components/SentimentBar'
import { Button, ErrorNote } from '../components/ui'
import { api } from '../lib/api'
import { downloadText, extractReviews, summaryToCsv } from '../lib/csv'
import { aspectCategoryLabel, seconds } from '../lib/format'
import { useI18n, type TFn, type UiLang } from '../lib/i18n'
import { INSIGHTS_SAMPLE } from '../lib/samples'
import type { BatchResponse, SummaryRow } from '../lib/types'

const MAX_REVIEWS = 200
type Sort = 'mentions' | 'worst' | 'best'

export function rowLabel(row: SummaryRow, lang: UiLang): string {
  return row.is_category ? aspectCategoryLabel(row.key.replace(/^cat:/, ''), lang) : row.label
}

export function headline(rows: SummaryRow[], lang: UiLang, t: TFn): string {
  const minMentions = rows.length > 6 ? 2 : 1
  const pool = rows.filter((r) => r.mentions >= minMentions)
  const praised = pool.filter((r) => r.net > 0).sort((a, b) => b.positive - a.positive || b.net - a.net).slice(0, 2)
  const worst = pool.filter((r) => r.net < 0).sort((a, b) => b.negative - a.negative || a.net - b.net)[0]
  const join = lang === 'vi' ? ' và ' : ' and '
  const pos = praised.map((r) => rowLabel(r, lang)).join(join)
  if (praised.length && worst) return t('insights.headBoth', { pos, neg: rowLabel(worst, lang), k: worst.negative, m: worst.mentions })
  if (praised.length) return t('insights.headPos', { pos })
  if (worst) return t('insights.headNeg', { neg: rowLabel(worst, lang), k: worst.negative, m: worst.mentions })
  return t('insights.headNone')
}

export function InsightsPage() {
  const { t } = useI18n()
  const [raw, setRaw] = useState('')
  const [fileName, setFileName] = useState('')
  const [sort, setSort] = useState<Sort>('mentions')
  const fileInput = useRef<HTMLInputElement>(null)
  const reviews = useMemo(() => extractReviews(raw, fileName), [raw, fileName])
  const batch = useMutation({ mutationFn: (texts: string[]) => api.batch(texts) })
  const data = batch.data

  const onFile = async (f: File | undefined) => {
    if (!f) return
    setFileName(f.name)
    setRaw(await f.text())
  }

  if (data) {
    return <InsightsResults data={data} sort={sort} setSort={setSort} source={fileName} onChange={() => batch.reset()} />
  }
  return (
    <>
      <div className="mb-7 max-w-[760px]">
        <h1 className="m-0 text-[28px] font-bold leading-tight tracking-[-0.015em]">{t('insights.title')}</h1>
        <p className="m-0 mt-1.5 text-ink-2">{t('insights.lead')}</p>
      </div>
      <form
        className="max-w-[880px] rounded-2xl border border-line bg-surface p-5 md:p-7"
        onSubmit={(e) => {
          e.preventDefault()
          if (reviews.length) batch.mutate(reviews.slice(0, MAX_REVIEWS))
        }}
      >
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <label htmlFor="reviews" className="font-semibold">
            {t('insights.input')}
          </label>
          <span className="text-[13px] text-ink-2" aria-live="polite">
            {t('insights.count', { n: reviews.length })}
            {reviews.length > MAX_REVIEWS && `. ${t('insights.tooMany')}`}
          </span>
        </div>
        <textarea
          id="reviews"
          rows={10}
          value={raw}
          onChange={(e) => {
            setRaw(e.target.value)
            setFileName('')
          }}
          placeholder={t('insights.placeholder')}
          className="block w-full resize-y rounded-xl border border-line-strong bg-paper px-4 py-3 font-serif text-base leading-relaxed text-ink placeholder:text-ink-3"
        />
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Button onClick={() => fileInput.current?.click()}>
            <Upload aria-hidden className="size-4" />
            {t('insights.upload')}
          </Button>
          <input ref={fileInput} type="file" accept=".csv,.txt,text/csv,text/plain" hidden onChange={(e) => onFile(e.target.files?.[0])} />
          <Button
            variant="quiet"
            onClick={() => {
              setRaw(INSIGHTS_SAMPLE)
              setFileName('')
            }}
          >
            {t('insights.useSample')}
          </Button>
          <div className="flex-1" />
          <Button type="submit" variant="primary" size="lg" busy={batch.isPending} disabled={!reviews.length || batch.isPending}>
            {batch.isPending ? t('insights.submitting', { n: Math.min(reviews.length, MAX_REVIEWS) }) : t('insights.submit')}
          </Button>
        </div>
        {batch.isError && (
          <div className="mt-4">
            <ErrorNote error={batch.error} onRetry={() => batch.mutate(reviews.slice(0, MAX_REVIEWS))} />
          </div>
        )}
        {!reviews.length && <p className="m-0 mt-3 text-[13px] text-ink-2">{t('insights.empty')}</p>}
      </form>
    </>
  )
}

function InsightsResults({ data, sort, setSort, source, onChange }: { data: BatchResponse; sort: Sort; setSort: (s: Sort) => void; source: string; onChange: () => void }) {
  const { t, lang } = useI18n()
  const rows = useMemo(() => {
    const r = [...data.summary]
    if (sort === 'worst') r.sort((a, b) => a.net - b.net || b.mentions - a.mentions)
    if (sort === 'best') r.sort((a, b) => b.net - a.net || b.mentions - a.mentions)
    return r
  }, [data.summary, sort])
  const urgent = data.reviews.filter((r) => r.priority.level === 'high')
  const shown = rows.slice(0, 12)
  return (
    <>
      <div className="mb-5 flex flex-wrap items-end gap-4">
        <div className="min-w-0 flex-[1_1_420px]">
          <h1 className="m-0 text-[28px] font-bold leading-tight tracking-[-0.015em]">{t('insights.title')}</h1>
          <p className="m-0 mt-1.5 text-ink-2">
            {t('insights.resultsSub', { n: data.reviews.length, s: seconds(data.total_ms, lang) })}
            {source && ` ${source}`}
          </p>
        </div>
        <div className="flex flex-wrap gap-2.5">
          <Button onClick={onChange}>{t('insights.change')}</Button>
          <Button variant="primary" onClick={() => downloadText(summaryToCsv(rows, (r) => rowLabel(r, lang)), 'reviewlens-insights.csv')}>
            <Download aria-hidden className="size-4" />
            {t('insights.export')}
          </Button>
        </div>
      </div>
      <p className="m-0 mb-7 max-w-[860px] font-serif text-xl leading-relaxed md:text-[22px]">{headline(data.summary, lang, t)}</p>
      <div className="flex flex-wrap items-start gap-8">
        <section className="min-w-0 flex-[8_1_640px] rounded-2xl border border-line bg-surface pb-1 pt-2">
          <div className="flex flex-wrap items-center gap-3 px-5 py-3.5">
            <h2 className="m-0 flex-1 text-lg font-bold">{t('insights.tableTitle', { n: shown.length })}</h2>
            <label htmlFor="sort" className="text-[13px] text-ink-2">
              {t('insights.sort')}
            </label>
            <select id="sort" value={sort} onChange={(e) => setSort(e.target.value as Sort)} className="min-h-10 rounded-[10px] border border-line-strong bg-surface px-2.5 text-sm text-ink">
              <option value="mentions">{t('insights.sortMentions')}</option>
              <option value="worst">{t('insights.sortWorst')}</option>
              <option value="best">{t('insights.sortBest')}</option>
            </select>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px] border-collapse text-sm">
              <thead>
                <tr className="text-left text-[13px] text-ink-2">
                  <th scope="col" className="border-b border-line px-5 py-2.5 font-medium">{t('insights.colAspect')}</th>
                  <th scope="col" className="border-b border-line px-3 py-2.5 text-right font-medium">{t('insights.colMentions')}</th>
                  <th scope="col" className="w-[34%] border-b border-line px-3 py-2.5 font-medium">{t('insights.colDist')}</th>
                  <th scope="col" className="border-b border-line px-3 py-2.5 text-right font-medium">{t('insights.colNet')}</th>
                  <th scope="col" className="border-b border-line py-2.5 pl-3 pr-5 font-medium">{t('insights.colExample')}</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => (
                  <tr key={r.key} className={r.net <= -50 ? 'bg-neg-soft' : ''}>
                    <th scope="row" className="border-b border-sunken px-5 py-3 text-left font-semibold">{rowLabel(r, lang)}</th>
                    <td className="border-b border-sunken px-3 py-3 text-right">{r.mentions}</td>
                    <td className="border-b border-sunken px-3 py-3">
                      <CountBar positive={r.positive} neutral={r.neutral} negative={r.negative} label={t('insights.distAria', { pos: r.positive, neu: r.neutral, neg: r.negative })} />
                    </td>
                    <td className={`border-b border-sunken px-3 py-3 text-right font-semibold ${r.net > 0 ? 'text-pos-ink' : r.net < 0 ? 'text-neg-ink' : 'text-ink-2'}`}>
                      {r.net > 0 ? `+${r.net}` : r.net < 0 ? `−${Math.abs(r.net)}` : '0'}
                    </td>
                    <td className="max-w-[220px] truncate border-b border-sunken py-3 pl-3 pr-5 font-serif text-ink-2" title={r.example}>
                      “{r.example}”
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="m-0 px-5 pb-3.5 pt-3 text-[13px] text-ink-2">{t('insights.netNote')}</p>
        </section>
        <aside className="min-w-0 flex-[4_1_320px]" aria-labelledby="urgent-title">
          <div className="mb-3 mt-1 flex items-baseline justify-between gap-3">
            <h2 id="urgent-title" className="m-0 text-lg font-bold">{t('insights.supportTitle')}</h2>
            <span className="text-[13px] text-ink-2">{t('insights.supportCount', { n: urgent.length })}</span>
          </div>
          {urgent.length === 0 && <p className="m-0 rounded-xl border border-dashed border-line-strong bg-surface p-4 text-sm text-ink-2">{t('insights.supportEmpty')}</p>}
          <ul className="m-0 flex list-none flex-col gap-2.5 p-0">
            {urgent.slice(0, 6).map((r) => (
              <li key={r.index} className="rounded-xl border border-line bg-surface p-4">
                <div className="mb-1.5 flex justify-between gap-2">
                  <span className="rounded-full bg-neg-mark px-2.5 py-0.5 text-xs font-bold text-neg-ink">{t('priority.high')}</span>
                  <span className="text-xs text-ink-2">{t('insights.review', { n: r.index + 1 })}</span>
                </div>
                <p className="m-0 mb-2.5 line-clamp-3 font-serif text-[15px] leading-relaxed">“{r.text}”</p>
                <div className="flex flex-wrap gap-1.5 text-xs">
                  {r.aspects
                    .filter((a) => a.sentiment === 'negative')
                    .map((a) => (
                      <span key={`${a.start}`} className="rounded-md bg-neg-mark px-2 py-0.5 text-neg-ink">
                        {a.category ? aspectCategoryLabel(a.category, lang) : a.term}
                      </span>
                    ))}
                </div>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </>
  )
}
