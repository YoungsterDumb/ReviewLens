import { useQuery } from '@tanstack/react-query'
import { ExternalLink } from 'lucide-react'
import { ErrorNote, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { num, pct } from '../lib/format'
import { useI18n, type TKey, type UiLang } from '../lib/i18n'
import type { Metric } from '../lib/types'

const PERCENT_METRICS = new Set(['long_context_coverage', 'vqa_accuracy', 'xai_fidelity'])
const REPO_URL = import.meta.env.VITE_REPO_URL as string | undefined

export function formatMetric(key: string, value: number, lang: UiLang): string {
  if (key === 'vae_recon_mse') return num(value, lang, 4)
  if (PERCENT_METRICS.has(key)) return pct(value, lang, value < 0.995 ? 1 : 0)
  return num(value, lang, 2)
}

/** Position on the bar: higher-is-better metrics use [0, 1]; lower-is-better uses [0, 2 x target]. */
export function barShare(m: Metric, value: number): number {
  const scale = m.direction === 'lower' ? m.target * 2 : Math.max(1, m.target)
  return Math.max(0, Math.min(1, value / scale))
}

function MetricRow({ m }: { m: Metric }) {
  const { t, lang } = useI18n()
  const status = m.met == null ? 'pending' : m.met ? 'met' : 'notMet'
  const tone = { met: 'text-pos-ink', notMet: 'text-neg-ink', pending: 'text-ink-2' }[status]
  const details = Object.entries(m.details).filter(([, v]) => v != null)
  return (
    <div className="grid grid-cols-1 items-center gap-x-7 gap-y-3 border-b border-sunken py-5 last:border-b-0 md:grid-cols-[minmax(200px,1.2fr)_minmax(240px,2fr)_120px]">
      <div>
        <p className="m-0 font-semibold">{t(`metric.${m.key}` as TKey)}</p>
        <p className="m-0 mt-0.5 text-[13px] text-ink-2">{t(`metric.${m.key}.desc` as TKey)}</p>
        {details.length > 0 && (
          <p className="m-0 mt-1 text-xs text-ink-2">
            {details.map(([k, v]) => `${t(`detail.${k}` as TKey)} ${typeof v === 'number' ? (Number.isInteger(v) && v > 1 ? num(v, lang, 0) : num(v, lang, 4)) : v}`).join(', ')}
          </p>
        )}
      </div>
      <div className="relative mt-4 h-[22px] md:mt-0">
        <div className="absolute inset-x-0 top-[7px] h-2 rounded-full bg-sunken" />
        {m.value != null && (
          <div
            className={`absolute left-0 top-[7px] h-2 rounded-full ${m.met === false ? 'bg-neg' : 'bg-ink'}`}
            style={{ width: `${barShare(m, m.value) * 100}%` }}
          />
        )}
        <div className="absolute inset-y-0 w-0.5 bg-neg-ink" style={{ left: `${barShare(m, m.target) * 100}%` }} />
        <span className="absolute -top-[18px] -translate-x-1/2 whitespace-nowrap text-xs text-ink-2" style={{ left: `${barShare(m, m.target) * 100}%` }}>
          {t('models.target', { v: formatMetric(m.key, m.target, lang) })}
        </span>
      </div>
      <div className="md:text-right">
        <p className="m-0 text-[22px] font-bold">{m.value != null ? formatMetric(m.key, m.value, lang) : '—'}</p>
        <p className={`m-0 text-[13px] font-semibold ${tone}`}>{t(`models.${status}` as TKey)}</p>
      </div>
    </div>
  )
}

const MODEL_ROWS: { id: string; task: TKey; data: TKey }[] = [
  { id: 'ate', task: 'task.ate', data: 'data.absa' },
  { id: 'atsc', task: 'task.atsc', data: 'data.absa' },
  { id: 'longctx', task: 'task.longctx', data: 'data.longctx' },
  { id: 'cvae', task: 'task.cvae', data: 'data.cvae' },
  { id: 'whisper', task: 'task.whisper', data: 'data.pretrained' },
  { id: 'blip', task: 'task.blip', data: 'data.vqa' },
  { id: 'clip', task: 'task.clip', data: 'data.pretrained' },
]

export function ModelsPage() {
  const { t, lang } = useI18n()
  const q = useQuery({ queryKey: ['metrics'], queryFn: api.metrics, staleTime: 5 * 60_000 })
  const metrics = q.data?.metrics ?? []
  const measured = metrics.filter((m) => m.value != null)
  const met = metrics.filter((m) => m.met).length
  const lead = !q.data
    ? ''
    : measured.length === metrics.length
      ? t('models.leadMet', { met, total: metrics.length })
      : t('models.leadPending', { done: measured.length, total: metrics.length })
  const updated = q.data?.updated_at
    ? t('models.updated', {
        date: new Intl.DateTimeFormat(lang === 'vi' ? 'vi-VN' : 'en-US', { dateStyle: 'medium' }).format(new Date(q.data.updated_at)),
        source: q.data.source === 'hub' ? t('models.sourceHub') : t('models.sourceLocal'),
      })
    : ''
  return (
    <>
      <div className="mb-7 flex flex-wrap items-end gap-4">
        <div className="min-w-0 flex-[1_1_420px]">
          <h1 className="m-0 text-[28px] font-bold leading-tight tracking-[-0.015em]">{t('models.title')}</h1>
          {q.data && <p className="m-0 mt-1.5 text-ink-2">{`${lead} ${updated}`.trim()}</p>}
        </div>
        {REPO_URL && (
          <a href={`${REPO_URL}/tree/main/training/notebooks`} target="_blank" rel="noreferrer" className="font-semibold text-ink">
            {t('models.notebooks')}
          </a>
        )}
      </div>
      {q.isPending && <Spinner label="…" />}
      {q.isError && <ErrorNote error={q.error} onRetry={() => q.refetch()} />}
      {q.data && (
        <>
          <section aria-label={t('models.title')} className="rounded-2xl border border-line bg-surface px-5 py-2 md:px-7">
            {metrics.map((m) => (
              <MetricRow key={m.key} m={m} />
            ))}
          </section>
          <section className="mt-10">
            <h2 className="m-0 mb-3.5 text-lg font-bold">{t('models.tableTitle')}</h2>
            <div className="overflow-x-auto rounded-2xl border border-line bg-surface">
              <table className="w-full min-w-[720px] border-collapse text-sm">
                <thead>
                  <tr className="text-left text-[13px] text-ink-2">
                    <th scope="col" className="border-b border-line px-5 py-3 font-medium">{t('models.colTask')}</th>
                    <th scope="col" className="border-b border-line px-3 py-3 font-medium">{t('models.colModel')}</th>
                    <th scope="col" className="border-b border-line px-3 py-3 font-medium">{t('models.colData')}</th>
                    <th scope="col" className="border-b border-line py-3 pl-3 pr-5 font-medium">{t('models.colSource')}</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_ROWS.map((row) => {
                    const id = q.data.models[row.id] ?? ''
                    const onHub = id.includes('/') && !id.includes(' ')
                    return (
                      <tr key={row.id}>
                        <td className="border-b border-sunken px-5 py-3 font-semibold">{t(row.task)}</td>
                        <td className="border-b border-sunken px-3 py-3 font-mono text-[13px]">{id}</td>
                        <td className="border-b border-sunken px-3 py-3">{t(row.data)}</td>
                        <td className="border-b border-sunken py-3 pl-3 pr-5">
                          {onHub && (
                            <a href={`https://huggingface.co/${id}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-ink">
                              Hugging Face
                              <ExternalLink aria-hidden className="size-3.5" />
                            </a>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </>
  )
}
