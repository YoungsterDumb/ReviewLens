import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { ArrowDownRight, ArrowRight, ArrowUpRight, LoaderCircle } from 'lucide-react'
import { ApiError } from '../lib/api'
import { useI18n } from '../lib/i18n'
import type { Sentiment } from '../lib/types'

type Variant = 'primary' | 'secondary' | 'quiet'

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-primary text-on-primary hover:opacity-90 border border-transparent',
  secondary: 'bg-surface text-ink border border-line-strong hover:bg-sunken',
  quiet: 'bg-transparent text-ink border border-transparent hover:bg-sunken',
}

export function Button({
  variant = 'secondary',
  size = 'md',
  busy = false,
  className = '',
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' | 'lg'; busy?: boolean }) {
  const sizes = { sm: 'min-h-8 px-2.5 text-[13px] rounded-lg', md: 'min-h-11 px-4 rounded-[10px]', lg: 'min-h-12 px-7 text-base rounded-[10px]' }
  return (
    <button
      type="button"
      {...rest}
      aria-busy={busy || undefined}
      className={`inline-flex items-center justify-center gap-2 font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-55 cursor-pointer ${sizes[size]} ${VARIANTS[variant]} ${className}`}
    >
      {busy && <LoaderCircle aria-hidden className="size-4 animate-spin" />}
      {children}
    </button>
  )
}

export function Spinner({ label }: { label: string }) {
  return (
    <span role="status" className="inline-flex items-center gap-2 text-[13px] text-ink-2">
      <LoaderCircle aria-hidden className="size-4 animate-spin" />
      {label}
    </span>
  )
}

const SENT_STYLE: Record<Sentiment, { text: string; mark: string; Icon: typeof ArrowUpRight }> = {
  positive: { text: 'text-pos-ink', mark: 'bg-pos-mark', Icon: ArrowUpRight },
  negative: { text: 'text-neg-ink', mark: 'bg-neg-mark', Icon: ArrowDownRight },
  neutral: { text: 'text-neu-ink', mark: 'bg-neu-mark', Icon: ArrowRight },
}

export function sentimentStyle(s: Sentiment) {
  return SENT_STYLE[s]
}

export function SentimentLabel({ sentiment, confidence }: { sentiment: Sentiment; confidence?: number }) {
  const { t, lang } = useI18n()
  const { text, Icon } = SENT_STYLE[sentiment]
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap text-[13px] font-semibold ${text}`}>
      <Icon aria-hidden className="size-3.5" strokeWidth={2.4} />
      {t(`label.${sentiment}`)}
      {confidence != null && ` ${new Intl.NumberFormat(lang === 'vi' ? 'vi-VN' : 'en-US', { style: 'percent' }).format(confidence)}`}
    </span>
  )
}

export function NumberBadge({ n, sentiment }: { n: number; sentiment: Sentiment }) {
  const { text, mark } = SENT_STYLE[sentiment]
  return (
    <span aria-hidden className={`inline-flex size-[22px] flex-none items-center justify-center rounded-full text-xs font-bold ${mark} ${text}`}>
      {n}
    </span>
  )
}

export function ErrorNote({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const { t } = useI18n()
  let message = error instanceof Error ? error.message : String(error)
  if (error instanceof ApiError) {
    const key = `error.${error.code}` as const
    const known = ['error.network', 'error.timeout', 'error.model_unavailable', 'error.empty_input'] as const
    if ((known as readonly string[]).includes(key)) message = t(key as (typeof known)[number])
  }
  return (
    <div role="alert" className="flex flex-wrap items-center gap-3 rounded-xl border border-neg/40 bg-neg-soft px-4 py-3 text-sm text-neg-ink">
      <span className="flex-1">{message}</span>
      {onRetry && (
        <Button size="sm" onClick={onRetry}>
          {t('error.retry')}
        </Button>
      )}
    </div>
  )
}

export function SectionTitle({ children, aside, id }: { children: ReactNode; aside?: ReactNode; id?: string }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <h2 id={id} className="m-0 text-lg font-bold">
        {children}
      </h2>
      {aside && <span className="text-[13px] text-ink-2">{aside}</span>}
    </div>
  )
}
