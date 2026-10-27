import { Fragment } from 'react'
import type { Aspect } from '../lib/types'
import { sentimentStyle } from './ui'

/** The review as reading text, each aspect marked with its sentiment colour and a footnote number. */
export function ReviewText({ text, aspects, active, className = '' }: {
  text: string
  aspects: Aspect[]
  active?: number | null
  className?: string
}) {
  const ordered = aspects
    .map((a, i) => ({ a, i }))
    .sort((x, y) => x.a.start - y.a.start)
    .filter(({ a }, k, arr) => k === 0 || a.start >= arr[k - 1].a.end)
  const parts: React.ReactNode[] = []
  let cursor = 0
  for (const { a, i } of ordered) {
    if (a.start > cursor) parts.push(<Fragment key={`t${cursor}`}>{text.slice(cursor, a.start)}</Fragment>)
    const style = sentimentStyle(a.sentiment)
    parts.push(
      <Fragment key={`a${i}`}>
        <mark className={`mark-${a.sentiment} ${active === i ? 'mark-active' : ''}`} data-aspect={i}>
          {text.slice(a.start, a.end)}
        </mark>
        <sup className={`ml-px font-sans text-[11px] font-bold ${style.text}`}>{i + 1}</sup>
      </Fragment>,
    )
    cursor = a.end
  }
  if (cursor < text.length) parts.push(<Fragment key={`t${cursor}`}>{text.slice(cursor)}</Fragment>)
  return <p className={`m-0 whitespace-pre-line font-serif text-[19px] leading-[1.8] md:text-[21px] ${className}`}>{parts}</p>
}
