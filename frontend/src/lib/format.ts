import type { TFn, UiLang } from './i18n'
import type { Overall, Priority, Probs, Sentiment } from './types'

const locale = (lang: UiLang) => (lang === 'vi' ? 'vi-VN' : 'en-US')

export function pct(value: number, lang: UiLang, digits = 0): string {
  return new Intl.NumberFormat(locale(lang), { style: 'percent', maximumFractionDigits: digits }).format(value)
}

export function num(value: number, lang: UiLang, digits = 2): string {
  return new Intl.NumberFormat(locale(lang), { maximumFractionDigits: digits }).format(value)
}

export function seconds(ms: number, lang: UiLang): string {
  return num(ms / 1000, lang, ms < 10_000 ? 2 : 1)
}

export function clock(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export function signed(value: number, lang: UiLang, digits = 2): string {
  const formatted = num(Math.abs(value), lang, digits)
  return value > 0 ? `+${formatted}` : value < 0 ? `−${formatted}` : formatted
}

/** Shares that add up to exactly 100 (largest remainder), for labels under a stacked bar. */
export function roundedShares(probs: Probs): Probs {
  const keys: Sentiment[] = ['positive', 'neutral', 'negative']
  const total = keys.reduce((s, k) => s + Math.max(0, probs[k]), 0)
  if (total <= 0) return { positive: 0, neutral: 0, negative: 0 }
  const raw = keys.map((k) => (Math.max(0, probs[k]) / total) * 100)
  const floors = raw.map(Math.floor)
  let left = 100 - floors.reduce((a, b) => a + b, 0)
  const order = raw.map((r, i) => [r - floors[i], i] as const).sort((a, b) => b[0] - a[0])
  for (const [, i] of order) {
    if (left-- <= 0) break
    floors[i] += 1
  }
  return { positive: floors[0], neutral: floors[1], negative: floors[2] }
}

export function overallLabel(o: Overall, t: TFn): string {
  if (o.mixed && o.label === 'positive') return t('overall.mixedPositive')
  if (o.mixed && o.label === 'negative') return t('overall.mixedNegative')
  return t(`label.${o.label}`)
}

export function reasonText(reason: string, t: TFn): string {
  const [code, arg] = reason.split(':')
  if (code === 'negative_aspects') return t('reason.negative_aspects', { n: arg ?? '' })
  if (code === 'photo_defective') return t('reason.photo_defective')
  if (code === 'toxic_language') return t('reason.toxic_language')
  return reason
}

export function priorityReasons(p: Priority, t: TFn): string {
  if (!p.reasons.length) return t('reason.none')
  const parts = p.reasons.map((r) => reasonText(r, t))
  return `${parts.join(', ')}.`
}

const VI_ANSWERS: Record<string, string> = {
  yes: 'có', no: 'không', black: 'đen', white: 'trắng', gray: 'xám', grey: 'xám', red: 'đỏ', blue: 'xanh dương',
  green: 'xanh lá', yellow: 'vàng', brown: 'nâu', pink: 'hồng', purple: 'tím', orange: 'cam', silver: 'bạc',
  shoe: 'giày', shoes: 'giày', sneaker: 'giày thể thao', bag: 'túi', shirt: 'áo sơ mi', dress: 'váy', phone: 'điện thoại',
  laptop: 'máy tính xách tay', boot: 'bốt', boots: 'bốt', sandal: 'dép sandal', jacket: 'áo khoác', coat: 'áo khoác',
}
const VI_QUESTIONS: Record<string, string> = {
  'What is this?': 'Đây là gì?',
  'What color is it?': 'Màu gì?',
  'Is it damaged?': 'Có bị hỏng không?',
}

export function questionText(q: string, lang: UiLang): string {
  return lang === 'vi' ? VI_QUESTIONS[q] ?? q : q
}

export function answerText(a: string, lang: UiLang): string {
  return lang === 'vi' ? VI_ANSWERS[a.trim().toLowerCase()] ?? a : a
}

export const CATEGORY_LABELS_VI: Record<string, string> = {
  SCREEN: 'Màn hình', CAMERA: 'Camera', FEATURES: 'Tính năng', BATTERY: 'Pin', PERFORMANCE: 'Hiệu năng',
  STORAGE: 'Bộ nhớ', DESIGN: 'Thiết kế', PRICE: 'Giá', GENERAL: 'Tổng thể', 'SER&ACC': 'Dịch vụ & phụ kiện',
}
export const CATEGORY_LABELS_EN: Record<string, string> = {
  SCREEN: 'Screen', CAMERA: 'Camera', FEATURES: 'Features', BATTERY: 'Battery', PERFORMANCE: 'Performance',
  STORAGE: 'Storage', DESIGN: 'Design', PRICE: 'Price', GENERAL: 'General', 'SER&ACC': 'Service & accessories',
}

export function aspectCategoryLabel(code: string, lang: UiLang): string {
  return (lang === 'vi' ? CATEGORY_LABELS_VI : CATEGORY_LABELS_EN)[code] ?? code
}
