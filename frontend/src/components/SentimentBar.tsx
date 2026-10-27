import { roundedShares } from '../lib/format'
import { useI18n } from '../lib/i18n'
import type { Probs } from '../lib/types'

export function SentimentBar({ probs, height = 10, legend = true }: { probs: Probs; height?: number; legend?: boolean }) {
  const { t } = useI18n()
  const s = roundedShares(probs)
  return (
    <div>
      <div
        role="img"
        aria-label={t('overall.aria', { pos: `${s.positive}%`, neu: `${s.neutral}%`, neg: `${s.negative}%` })}
        className="flex gap-0.5 overflow-hidden rounded-full"
        style={{ height }}
      >
        {s.positive > 0 && <span className="bg-pos" style={{ width: `${s.positive}%` }} />}
        {s.neutral > 0 && <span className="bg-neu" style={{ width: `${s.neutral}%` }} />}
        {s.negative > 0 && <span className="bg-neg" style={{ width: `${s.negative}%` }} />}
      </div>
      {legend && (
        <div className="mt-2 flex flex-wrap gap-4 text-[13px] text-ink-2">
          <span>
            <b className="font-semibold text-pos-ink">{s.positive}%</b> {t('word.positive')}
          </span>
          <span>
            <b className="font-semibold text-neu-ink">{s.neutral}%</b> {t('word.neutral')}
          </span>
          <span>
            <b className="font-semibold text-neg-ink">{s.negative}%</b> {t('word.negative')}
          </span>
        </div>
      )}
    </div>
  )
}

export function CountBar({ positive, neutral, negative, label }: { positive: number; neutral: number; negative: number; label: string }) {
  const total = positive + neutral + negative || 1
  return (
    <div role="img" aria-label={label} className="flex h-3 gap-0.5 overflow-hidden rounded-[3px]">
      {positive > 0 && <span className="bg-pos" style={{ width: `${(positive / total) * 100}%` }} />}
      {neutral > 0 && <span className="bg-neu" style={{ width: `${(neutral / total) * 100}%` }} />}
      {negative > 0 && <span className="bg-neg" style={{ width: `${(negative / total) * 100}%` }} />}
    </div>
  )
}
