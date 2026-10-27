import { useMutation } from '@tanstack/react-query'
import { useSearch } from '@tanstack/react-router'
import { Download, Shuffle } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Button, ErrorNote } from '../components/ui'
import { api } from '../lib/api'
import { useI18n, type TKey } from '../lib/i18n'
import type { Probs } from '../lib/types'

const COUNTS = [4, 8, 16] as const

/** Slider value in [-100, 100] -> sentiment mix with a fixed 10% neutral share. */
export function sliderToProbs(value: number): Probs {
  const t = (value + 100) / 200
  const positive = Math.round(t * 90) / 100
  const negative = Math.round((1 - t) * 90) / 100
  return { positive, negative, neutral: Math.round((1 - positive - negative) * 100) / 100 }
}

async function downloadGrid(images: string[], fileName: string) {
  const loaded = await Promise.all(
    images.map(
      (src) =>
        new Promise<HTMLImageElement>((resolve, reject) => {
          const img = new Image()
          img.onload = () => resolve(img)
          img.onerror = reject
          img.src = src
        }),
    ),
  )
  const cols = Math.min(4, loaded.length)
  const rows = Math.ceil(loaded.length / cols)
  const cell = loaded[0]?.width ?? 112
  const canvas = document.createElement('canvas')
  canvas.width = cols * cell
  canvas.height = rows * cell
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.imageSmoothingEnabled = false
  loaded.forEach((img, i) => ctx.drawImage(img, (i % cols) * cell, Math.floor(i / cols) * cell, cell, cell))
  const a = document.createElement('a')
  a.href = canvas.toDataURL('image/png')
  a.download = fileName
  a.click()
}

export function GeneratePage() {
  const { t } = useI18n()
  const search = useSearch({ from: '/generate' })
  const [category, setCategory] = useState<number>(search.category ?? 7)
  const [slider, setSlider] = useState(40)
  const [count, setCount] = useState<number>(8)
  const [seed, setSeed] = useState<number>(42)
  const probs = sliderToProbs(slider)
  const gen = useMutation({ mutationFn: () => api.generate(category, probs, count, seed, true) })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    gen.mutate()
  }
  const pctOf = (v: number) => `${Math.round(v * 100)}%`
  return (
    <>
      <div className="mb-7 max-w-[760px]">
        <h1 className="m-0 text-[28px] font-bold leading-tight tracking-[-0.015em]">{t('generate.title')}</h1>
        <p className="m-0 mt-1.5 text-ink-2">{t('generate.lead')}</p>
      </div>
      <div className="flex flex-wrap items-start gap-8">
        <form onSubmit={submit} className="min-w-0 flex-[4_1_320px] rounded-2xl border border-line bg-surface p-6">
          <fieldset className="m-0 mb-6 border-0 p-0">
            <legend className="mb-2.5 p-0 font-semibold">{t('generate.category')}</legend>
            <div className="grid grid-cols-2 gap-2">
              {Array.from({ length: 10 }, (_, i) => (
                <button
                  key={i}
                  type="button"
                  aria-pressed={category === i}
                  onClick={() => setCategory(i)}
                  className={`min-h-11 cursor-pointer rounded-[10px] px-3 text-left text-sm ${category === i ? 'border-2 border-primary bg-primary font-semibold text-on-primary' : 'border border-line-strong bg-surface text-ink hover:bg-sunken'}`}
                >
                  {t(`cat.${i}` as TKey)}
                </button>
              ))}
            </div>
          </fieldset>
          <div className="mb-6">
            <label htmlFor="sent" className="mb-2 block font-semibold">
              {t('generate.sentiment')}
            </label>
            <input
              id="sent"
              type="range"
              min={-100}
              max={100}
              step={5}
              value={slider}
              onChange={(e) => setSlider(Number(e.target.value))}
              aria-valuetext={t('generate.mix', { pos: pctOf(probs.positive), neu: pctOf(probs.neutral), neg: pctOf(probs.negative) })}
              className="h-7 w-full accent-[var(--ink)]"
            />
            <div className="flex justify-between text-[13px] text-ink-2">
              <span>{t('label.negative')}</span>
              <span>{t('label.positive')}</span>
            </div>
            <p className="m-0 mt-2 text-sm">
              <b className="font-semibold text-pos-ink">{pctOf(probs.positive)}</b> {t('word.positive')},{' '}
              <b className="font-semibold text-neu-ink">{pctOf(probs.neutral)}</b> {t('word.neutral')},{' '}
              <b className="font-semibold text-neg-ink">{pctOf(probs.negative)}</b> {t('word.negative')}
            </p>
          </div>
          <div className="mb-6 flex flex-wrap gap-4">
            <div className="flex-[1_1_140px]">
              <span id="count-label" className="mb-2 block font-semibold">
                {t('generate.count')}
              </span>
              <div role="group" aria-labelledby="count-label" className="inline-flex overflow-hidden rounded-[10px] border border-line-strong">
                {COUNTS.map((c) => (
                  <button
                    key={c}
                    type="button"
                    aria-pressed={count === c}
                    onClick={() => setCount(c)}
                    className={`min-h-10 min-w-12 cursor-pointer font-semibold ${count === c ? 'bg-primary text-on-primary' : 'bg-surface text-ink-2 hover:bg-sunken'}`}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex-[1_1_150px]">
              <label htmlFor="seed" className="mb-2 block font-semibold">
                {t('generate.seed')}
              </label>
              <div className="flex gap-2">
                <input
                  id="seed"
                  type="number"
                  min={0}
                  value={seed}
                  onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
                  className="min-h-10 w-full min-w-0 rounded-[10px] border border-line-strong bg-surface px-2.5 text-ink"
                />
                <button
                  type="button"
                  aria-label={t('generate.random')}
                  title={t('generate.random')}
                  onClick={() => setSeed(Math.floor(Math.random() * 1_000_000))}
                  className="inline-flex size-10 flex-none cursor-pointer items-center justify-center rounded-[10px] border border-line-strong bg-surface text-ink hover:bg-sunken"
                >
                  <Shuffle aria-hidden className="size-4" />
                </button>
              </div>
            </div>
          </div>
          <Button type="submit" variant="primary" size="lg" className="w-full" busy={gen.isPending}>
            {gen.isPending ? t('generate.submitting') : t('generate.submit')}
          </Button>
        </form>

        <section className="min-w-0 flex-[8_1_560px]" aria-live="polite">
          {gen.isError && <ErrorNote error={gen.error} onRetry={() => gen.mutate()} />}
          {!gen.data && !gen.isError && (
            <p className="m-0 rounded-2xl border border-dashed border-line-strong p-8 text-center text-ink-2">{t('generate.empty')}</p>
          )}
          {gen.data && (
            <>
              <div className="mb-3.5 flex flex-wrap items-center gap-3">
                <div className="flex-1">
                  <h2 className="m-0 text-lg font-bold">{t('generate.results', { n: gen.data.images.length })}</h2>
                  <p className="m-0 mt-0.5 text-[13px] text-ink-2">
                    {t('generate.resultsSub', { cat: t(`cat.${gen.data.category_index}` as TKey), seed: gen.data.seed })}
                  </p>
                </div>
                <Button onClick={() => downloadGrid(gen.data.images, `reviewlens-${gen.data.category.replace(/\W+/g, '-').toLowerCase()}-${gen.data.seed}.png`)}>
                  <Download aria-hidden className="size-4" />
                  {t('generate.download')}
                </Button>
              </div>
              <div className="grid grid-cols-4 gap-3">
                {gen.data.images.map((src, i) => (
                  <img key={i} src={src} alt={t('variants.alt', { n: i + 1 })} className="pixelated aspect-square w-full rounded-[10px] bg-black" />
                ))}
              </div>
              {gen.data.sweep.length > 0 && (
                <div className="mt-8 border-t border-line-strong pt-5">
                  <h2 className="m-0 mb-1 text-base font-bold">{t('generate.sweepTitle')}</h2>
                  <p className="m-0 mb-3.5 text-[13px] text-ink-2">{t('generate.sweepLead')}</p>
                  <div className="grid grid-cols-7 gap-2">
                    {gen.data.sweep.map((src, i) => (
                      <img key={i} src={src} alt={t('generate.sweepAlt', { n: i + 1 })} className="pixelated aspect-square w-full rounded-lg bg-black" />
                    ))}
                  </div>
                  <div className="mt-1.5 flex justify-between text-[13px] text-ink-2">
                    <span>{t('label.negative')}</span>
                    <span>{t('label.neutral')}</span>
                    <span>{t('label.positive')}</span>
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </>
  )
}
