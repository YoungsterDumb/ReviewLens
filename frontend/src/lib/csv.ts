import type { SummaryRow } from './types'

const TEXT_COLUMNS = ['review', 'reviews', 'text', 'comment', 'content', 'đánh giá', 'danh gia', 'nhận xét', 'body']

/** Minimal RFC 4180 parser: quoted fields, escaped quotes, commas/newlines inside quotes, CRLF. */
export function parseCsv(input: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  const text = input.replace(/^﻿/, '')
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"'
        i++
      } else if (c === '"') {
        quoted = false
      } else {
        field += c
      }
    } else if (c === '"') {
      quoted = true
    } else if (c === ',') {
      row.push(field)
      field = ''
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else {
      field += c
    }
  }
  if (field !== '' || row.length) {
    row.push(field)
    rows.push(row)
  }
  return rows.filter((r) => r.some((f) => f.trim() !== ''))
}

/** Reviews from a CSV (a text-like column, else the longest column) or plain text (one per line). */
export function extractReviews(content: string, fileName = ''): string[] {
  const isCsv = fileName.toLowerCase().endsWith('.csv') || /^[^\n]*,[^\n]*\n/.test(content)
  if (!isCsv) {
    return content.split(/\r?\n/).map((l) => l.trim()).filter(Boolean)
  }
  const rows = parseCsv(content)
  if (!rows.length) return []
  const header = rows[0].map((h) => h.trim().toLowerCase())
  let col = header.findIndex((h) => TEXT_COLUMNS.includes(h))
  let body = rows.slice(1)
  if (col === -1) {
    body = rows
    const widths = header.map((_, i) => rows.reduce((s, r) => s + (r[i]?.length ?? 0), 0))
    col = widths.indexOf(Math.max(...widths))
  }
  return body.map((r) => (r[col] ?? '').trim()).filter(Boolean)
}

function escapeCsv(value: string | number): string {
  const s = String(value)
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function summaryToCsv(rows: SummaryRow[], labelOf: (r: SummaryRow) => string): string {
  const header = ['aspect', 'mentions', 'positive', 'neutral', 'negative', 'net_score', 'example']
  const lines = rows.map((r) => [labelOf(r), r.mentions, r.positive, r.neutral, r.negative, r.net, r.example].map(escapeCsv).join(','))
  return `﻿${header.join(',')}\n${lines.join('\n')}\n`
}

export function downloadText(content: string, fileName: string, type = 'text/csv;charset=utf-8'): void {
  const url = URL.createObjectURL(new Blob([content], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = fileName
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
