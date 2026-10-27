import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { api } from '../lib/api'
import { answerText, num, pct, questionText } from '../lib/format'
import { useI18n, type TKey } from '../lib/i18n'
import type { ImageResult, QA } from '../lib/types'
import { Button, ErrorNote, Spinner } from './ui'

export function PhotoPanel({ image, result }: { image: Blob; result: ImageResult }) {
  const { t, lang } = useI18n()
  const url = useMemo(() => URL.createObjectURL(image), [image])
  useEffect(() => () => URL.revokeObjectURL(url), [url])
  const [showInfluence, setShowInfluence] = useState(false)
  const explain = useQuery({
    queryKey: ['explain-image', url],
    queryFn: () => api.explainImage(image),
    enabled: showInfluence,
    staleTime: Infinity,
    retry: 1,
  })
  const [question, setQuestion] = useState('')
  const [asked, setAsked] = useState<QA[]>([])
  const vqa = useMutation({
    mutationFn: (q: string) => api.vqa(image, q),
    onSuccess: (r) => {
      setAsked((a) => [{ question: r.question, answer: r.answer }, ...a])
      setQuestion('')
    },
  })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (question.trim()) vqa.mutate(question.trim())
  }
  const defective = result.condition.defective >= result.condition.good
  const categoryLabel = result.category_index != null ? t(`cat.${result.category_index}` as TKey) : t('photo.otherCategory')
  const answers = [...asked, ...result.answers]

  return (
    <section aria-labelledby="photo-title">
      <div className="mb-3.5 flex items-center justify-between gap-3">
        <h2 id="photo-title" className="m-0 text-lg font-bold">
          {t('photo.title')}
        </h2>
        <label className="inline-flex cursor-pointer items-center gap-2 text-[13px] text-ink-2">
          <input type="checkbox" checked={showInfluence} onChange={(e) => setShowInfluence(e.target.checked)} className="size-[18px] accent-[var(--ink)]" />
          {t('photo.influence')}
        </label>
      </div>
      <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden rounded-xl bg-sunken">
        <img
          src={showInfluence && explain.data ? explain.data.heatmap : url}
          alt={t('photo.alt')}
          className="max-h-full max-w-full object-contain"
        />
        {showInfluence && explain.isPending && (
          <span className="absolute inset-x-3 bottom-3 rounded-full bg-surface px-3 py-1">
            <Spinner label={t('photo.explaining')} />
          </span>
        )}
        {showInfluence && explain.data && (
          <span className="absolute bottom-3 left-3 rounded-full bg-surface px-2.5 py-1 text-xs font-semibold text-neg-ink">
            {explain.data.label === 'defective' ? t('photo.legendDefective') : t('photo.legendGood')}
            {', '}
            {t('explain.fidelity', { p: pct(explain.data.fidelity, lang) })}
          </span>
        )}
      </div>
      {showInfluence && explain.isError && (
        <div className="mt-3">
          <ErrorNote error={explain.error} onRetry={() => explain.refetch()} />
        </div>
      )}
      <dl className="mt-4 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
        <dt className="text-ink-2">{t('photo.category')}</dt>
        <dd className="m-0 font-semibold">
          {categoryLabel}, {pct(result.category_confidence, lang)}
        </dd>
        <dt className="text-ink-2">{t('photo.condition')}</dt>
        <dd className={`m-0 font-semibold ${defective ? 'text-neg-ink' : 'text-pos-ink'}`}>
          {defective ? t('photo.defective') : t('photo.good')}, {pct(defective ? result.condition.defective : result.condition.good, lang)}
        </dd>
        {result.aspect_alignment.length > 0 && (
          <>
            <dt className="text-ink-2">{t('photo.alignment')}</dt>
            <dd className="m-0">
              {result.aspect_alignment
                .slice()
                .sort((a, b) => b.score - a.score)
                .slice(0, 3)
                .map((a) => `${a.term} ${num(a.score, lang)}`)
                .join(', ')}
            </dd>
          </>
        )}
      </dl>
      <form className="mt-4" onSubmit={submit}>
        <label htmlFor="vqa-q" className="mb-1.5 block text-[13px] font-semibold">
          {t('photo.ask')}
        </label>
        <div className="flex gap-2">
          <input
            id="vqa-q"
            value={question}
            maxLength={200}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={t('photo.askPlaceholder')}
            aria-describedby="vqa-hint"
            className="min-h-11 min-w-0 flex-1 rounded-[10px] border border-line-strong bg-surface px-3 text-ink placeholder:text-ink-3"
          />
          <Button type="submit" variant="primary" busy={vqa.isPending} disabled={!question.trim()}>
            {t('photo.askButton')}
          </Button>
        </div>
        <p id="vqa-hint" className="m-0 mt-1.5 text-xs text-ink-2">
          {t('photo.askHint')}
        </p>
      </form>
      {vqa.isError && (
        <div className="mt-3">
          <ErrorNote error={vqa.error} />
        </div>
      )}
      <ul className="m-0 mt-2 flex list-none flex-col p-0 text-sm" aria-live="polite">
        {answers.map((qa, i) => (
          <li key={`${qa.question}-${i}`} className="flex justify-between gap-3 border-b border-line py-2 last:border-b-0">
            <span className="text-ink-2">{questionText(qa.question, lang)}</span>
            <span className="font-semibold">{answerText(qa.answer, lang)}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
