import { ImagePlus, Mic, Square, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { clock } from '../lib/format'
import { useI18n } from '../lib/i18n'

export const MAX_IMAGE_BYTES = 8 * 1024 * 1024
export const MAX_AUDIO_SECONDS = 120

function sizeLabel(bytes: number, lang: 'vi' | 'en') {
  const mb = bytes / (1024 * 1024)
  return `${new Intl.NumberFormat(lang === 'vi' ? 'vi-VN' : 'en-US', { maximumFractionDigits: 1 }).format(mb)} MB`
}

export function ImagePicker({ file, onChange }: { file: File | null; onChange: (f: File | null) => void }) {
  const { t, lang } = useI18n()
  const input = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url)
  }, [url])

  const accept = (f: File | undefined | null) => {
    if (!f) return
    if (!f.type.startsWith('image/')) return
    if (f.size > MAX_IMAGE_BYTES) {
      setError(t('analyze.photoTooBig'))
      return
    }
    setError(null)
    onChange(f)
  }

  if (file && url) {
    return (
      <div className="flex min-w-0 flex-1 basis-60 items-center gap-3.5 rounded-xl border border-line p-3">
        <img src={url} alt="" className="size-16 flex-none rounded-lg bg-sunken object-cover" />
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-sm font-semibold">{file.name}</p>
          <p className="m-0 text-[13px] text-ink-2">{sizeLabel(file.size, lang)}</p>
        </div>
        <button type="button" aria-label={t('analyze.photoRemove')} onClick={() => onChange(null)} className="inline-flex size-10 cursor-pointer items-center justify-center rounded-lg bg-sunken text-ink">
          <X aria-hidden className="size-4" />
        </button>
      </div>
    )
  }
  return (
    <div
      className={`flex min-w-0 flex-1 basis-60 flex-col justify-center rounded-xl border border-dashed p-3 ${dragging ? 'border-ink bg-sunken' : 'border-line-strong'}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        accept(e.dataTransfer.files?.[0])
      }}
    >
      <button type="button" onClick={() => input.current?.click()} className="flex cursor-pointer items-center gap-3.5 bg-transparent p-0 text-left text-ink">
        <span className="inline-flex size-16 flex-none items-center justify-center rounded-lg bg-sunken">
          <ImagePlus aria-hidden className="size-6" />
        </span>
        <span>
          <span className="block text-sm font-semibold">{t('analyze.photoAdd')}</span>
          <span className="block text-[13px] text-ink-2">{t('analyze.photoHint')}</span>
        </span>
      </button>
      <input ref={input} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(e) => accept(e.target.files?.[0])} />
      {error && <p role="alert" className="m-0 mt-2 text-[13px] text-neg-ink">{error}</p>}
    </div>
  )
}

export interface VoiceNote {
  blob: Blob
  seconds: number
  name: string
}

export function VoiceRecorder({ value, onChange }: { value: VoiceNote | null; onChange: (v: VoiceNote | null) => void }) {
  const { t } = useI18n()
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [denied, setDenied] = useState(false)
  const recorder = useRef<MediaRecorder | null>(null)
  const timer = useRef<number | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const supported = typeof window !== 'undefined' && 'MediaRecorder' in window && !!navigator.mediaDevices?.getUserMedia

  const stop = () => {
    if (timer.current) window.clearInterval(timer.current)
    timer.current = null
    if (recorder.current?.state === 'recording') recorder.current.stop()
    setRecording(false)
  }
  useEffect(() => () => stop(), [])

  const start = async () => {
    setDenied(false)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const chunks: BlobPart[] = []
      const rec = new MediaRecorder(stream)
      const began = Date.now()
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
      rec.onstop = () => {
        stream.getTracks().forEach((tr) => tr.stop())
        const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' })
        onChange({ blob, seconds: (Date.now() - began) / 1000, name: 'voice-note.webm' })
      }
      rec.start()
      recorder.current = rec
      setElapsed(0)
      setRecording(true)
      timer.current = window.setInterval(() => {
        const s = (Date.now() - began) / 1000
        setElapsed(s)
        if (s >= MAX_AUDIO_SECONDS) stop()
      }, 250)
    } catch {
      setDenied(true)
    }
  }

  const onFile = (f: File | undefined) => {
    if (!f) return
    const probe = new Audio()
    const url = URL.createObjectURL(f)
    probe.preload = 'metadata'
    probe.onloadedmetadata = () => {
      onChange({ blob: f, seconds: Number.isFinite(probe.duration) ? probe.duration : 0, name: f.name })
      URL.revokeObjectURL(url)
    }
    probe.onerror = () => {
      onChange({ blob: f, seconds: 0, name: f.name })
      URL.revokeObjectURL(url)
    }
    probe.src = url
  }

  const audioUrl = useMemo(() => (value ? URL.createObjectURL(value.blob) : null), [value])
  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl)
  }, [audioUrl])

  if (value && !recording && audioUrl) {
    return (
      <div className="flex min-w-0 flex-1 basis-60 items-center gap-3.5 rounded-xl border border-line p-3">
        <span className="inline-flex size-16 flex-none items-center justify-center rounded-full bg-sunken">
          <Mic aria-hidden className="size-6" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-sm font-semibold">{t('analyze.voiceReady', { time: clock(value.seconds) })}</p>
          <audio controls src={audioUrl} className="mt-1 h-8 w-full max-w-60" />
        </div>
        <button type="button" aria-label={t('analyze.voiceRemove')} onClick={() => onChange(null)} className="inline-flex size-10 cursor-pointer items-center justify-center rounded-lg bg-sunken text-ink">
          <X aria-hidden className="size-4" />
        </button>
      </div>
    )
  }
  return (
    <div className="flex min-w-0 flex-1 basis-60 items-center gap-3.5 rounded-xl border border-dashed border-line-strong p-3">
      <button
        type="button"
        aria-label={recording ? t('analyze.voiceStop') : t('analyze.voiceStart')}
        aria-pressed={recording}
        disabled={!supported}
        onClick={recording ? stop : start}
        className={`inline-flex size-16 flex-none cursor-pointer items-center justify-center rounded-full disabled:cursor-not-allowed disabled:opacity-50 ${recording ? 'bg-neg text-white' : 'bg-primary text-on-primary'}`}
      >
        {recording ? <Square aria-hidden className="size-5" fill="currentColor" /> : <Mic aria-hidden className="size-6" />}
      </button>
      <div className="min-w-0 flex-1">
        <p className="m-0 text-sm font-semibold" aria-live="polite">
          {recording ? t('analyze.voiceRecording', { time: `${clock(elapsed)} / ${clock(MAX_AUDIO_SECONDS)}` }) : t('analyze.voiceTitle')}
        </p>
        <p className="m-0 text-[13px] text-ink-2">
          <button type="button" onClick={() => fileInput.current?.click()} className="cursor-pointer bg-transparent p-0 text-[13px] text-ink underline">
            {t('analyze.voiceUpload')}
          </button>
          {t('analyze.voiceLimit')}
        </p>
        {denied && <p role="alert" className="m-0 mt-1 text-[13px] text-neg-ink">{t('analyze.voiceDenied')}</p>}
      </div>
      <input ref={fileInput} type="file" accept="audio/*" hidden onChange={(e) => onFile(e.target.files?.[0])} />
    </div>
  )
}
