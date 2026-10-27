import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, analyzeForm, api } from './api'
import { extractReviews, parseCsv, summaryToCsv } from './csv'
import { answerText, clock, overallLabel, priorityReasons, questionText, roundedShares, signed } from './format'
import { dictionaries, translate } from './i18n'
import { sliderToProbs } from '../routes/GeneratePage'
import { headline } from '../routes/InsightsPage'
import { barShare, formatMetric } from '../routes/ModelsPage'
import type { Metric, SummaryRow } from './types'

const tVi = (k: Parameters<typeof translate>[1], v?: Record<string, string | number>) => translate('vi', k, v)
const tEn = (k: Parameters<typeof translate>[1], v?: Record<string, string | number>) => translate('en', k, v)

describe('i18n', () => {
  it('has the same keys in Vietnamese and English, none empty', () => {
    expect(Object.keys(dictionaries.en).sort()).toEqual(Object.keys(dictionaries.vi).sort())
    for (const d of Object.values(dictionaries)) for (const v of Object.values(d)) expect(v.trim()).not.toBe('')
  })

  it('interpolates variables and keeps unknown ones', () => {
    expect(tVi('aspects.title', { n: 3 })).toBe('3 khía cạnh')
    expect(tEn('analyze.chars', { n: 5 })).toBe('5 / {max} characters')
  })

  it('uses the same placeholders in both languages', () => {
    const vars = (s: string) => (s.match(/\{\w+\}/g) ?? []).sort().join()
    for (const k of Object.keys(dictionaries.vi) as (keyof typeof dictionaries.vi)[]) {
      expect(vars(dictionaries.en[k]), k).toBe(vars(dictionaries.vi[k]))
    }
  })
})

describe('format', () => {
  it('rounds shares to exactly 100', () => {
    const s = roundedShares({ positive: 0.333, neutral: 0.333, negative: 0.334 })
    expect(s.positive + s.neutral + s.negative).toBe(100)
    expect(roundedShares({ positive: 0, neutral: 0, negative: 0 })).toEqual({ positive: 0, neutral: 0, negative: 0 })
  })

  it('labels mixed sentiment', () => {
    const probs = { positive: 0.6, neutral: 0.1, negative: 0.3 }
    expect(overallLabel({ label: 'positive', mixed: true, probs }, tVi)).toBe('Tích cực, kèm phàn nàn')
    expect(overallLabel({ label: 'negative', mixed: false, probs }, tEn)).toBe('Negative')
  })

  it('explains priority reasons', () => {
    expect(priorityReasons({ level: 'high', reasons: ['photo_defective', 'negative_aspects:2'] }, tVi)).toBe('Ảnh có dấu hiệu lỗi, 2 khía cạnh tiêu cực.')
    expect(priorityReasons({ level: 'low', reasons: [] }, tEn)).toBe('Nothing that needs urgent handling.')
  })

  it('formats clocks, signs and translated VQA text', () => {
    expect(clock(83.4)).toBe('1:23')
    expect(signed(0.42, 'vi')).toBe('+0,42')
    expect(signed(-0.07, 'en')).toBe('−0.07')
    expect(questionText('Is it damaged?', 'vi')).toBe('Có bị hỏng không?')
    expect(answerText('Gray', 'vi')).toBe('xám')
    expect(answerText('gray', 'en')).toBe('gray')
  })
})

describe('csv', () => {
  it('parses quotes, escaped quotes and newlines inside fields', () => {
    expect(parseCsv('a,b\n"x, y","he said ""hi""\nok"\r\n')).toEqual([
      ['a', 'b'],
      ['x, y', 'he said "hi"\nok'],
    ])
  })

  it('finds the review column by header, else uses the longest column', () => {
    expect(extractReviews('id,review\n1,Great screen\n2,"Bad, slow fan"\n', 'r.csv')).toEqual(['Great screen', 'Bad, slow fan'])
    expect(extractReviews('1,Great screen\n2,Pin trâu nhưng hơi nóng\n', 'r.csv')).toEqual(['Great screen', 'Pin trâu nhưng hơi nóng'])
    expect(extractReviews('one\n\n two \n', 'notes.txt')).toEqual(['one', 'two'])
  })

  it('exports a summary with escaping and a BOM for Excel', () => {
    const row: SummaryRow = { key: 'term:fan', label: 'fan', is_category: false, mentions: 2, positive: 0, neutral: 0, negative: 2, net: -100, example: 'fan, "loud"' }
    const csv = summaryToCsv([row], (r) => r.label)
    expect(csv.startsWith('﻿aspect,mentions')).toBe(true)
    expect(csv).toContain('fan,2,0,0,2,-100,"fan, ""loud"""')
  })
})

describe('page helpers', () => {
  it('maps the sentiment slider to a mix that sums to 1', () => {
    expect(sliderToProbs(100)).toEqual({ positive: 0.9, negative: 0, neutral: 0.1 })
    expect(sliderToProbs(-100)).toEqual({ positive: 0, negative: 0.9, neutral: 0.1 })
    const mid = sliderToProbs(0)
    expect(mid.positive + mid.negative + mid.neutral).toBeCloseTo(1)
  })

  it('writes the insights headline from praise and complaints', () => {
    const rows: SummaryRow[] = [
      { key: 'term:delivery', label: 'delivery', is_category: false, mentions: 5, positive: 5, neutral: 0, negative: 0, net: 100, example: '' },
      { key: 'cat:BATTERY', label: 'Pin', is_category: true, mentions: 4, positive: 1, neutral: 0, negative: 3, net: -50, example: '' },
    ]
    expect(headline(rows, 'en', tEn)).toBe('Customers praise delivery. The biggest complaint is Battery: 3 of 4 mentions are negative.')
    expect(headline(rows.slice(1), 'vi', tVi)).toBe('Phàn nàn lớn nhất là Pin: 3 trên 4 lượt nhắc là tiêu cực.')
    expect(headline([], 'en', tEn)).toBe('Not enough aspects to draw a conclusion yet.')
  })

  it('formats metrics and places them on the right scale', () => {
    const vae: Metric = { key: 'vae_recon_mse', value: 0.008, target: 0.01, direction: 'lower', met: true, details: {} }
    const f1: Metric = { key: 'ate_f1', value: 0.83, target: 0.8, direction: 'higher', met: true, details: {} }
    expect(formatMetric('vae_recon_mse', 0.0087, 'vi')).toBe('0,0087')
    expect(formatMetric('vqa_accuracy', 0.764, 'en')).toBe('76.4%')
    expect(barShare(vae, 0.01)).toBe(0.5)
    expect(barShare(f1, 0.83)).toBeCloseTo(0.83)
  })
})

describe('api client', () => {
  afterEach(() => vi.restoreAllMocks())

  it('builds the analyze form, skipping empty text', () => {
    const fd = analyzeForm({ text: '  ', lang: 'vi', image: new File(['x'], 'p.png'), audio: new Blob(['y']), seed: 3 })
    expect(fd.get('text')).toBeNull()
    expect(fd.get('lang')).toBe('vi')
    expect((fd.get('image') as File).name).toBe('p.png')
    expect((fd.get('audio') as File).name).toBe('voice-note.webm')
    expect(fd.get('seed')).toBe('3')
  })

  it('turns the error envelope into an ApiError', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ error: { code: 'empty_input', message: 'Add a review' } }), { status: 400 }),
    )
    await expect(api.analyze({ text: '' })).rejects.toMatchObject({ status: 400, code: 'empty_input' })
  })

  it('reports network failures', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'))
    const err = await api.health().catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.code).toBe('network')
  })
})
