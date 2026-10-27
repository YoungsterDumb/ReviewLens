import { Link, Outlet } from '@tanstack/react-router'
import { BarChart3, LayoutGrid, ListChecks, Moon, Search, Sun } from 'lucide-react'
import type { ReactNode } from 'react'
import { API_URL } from '../lib/api'
import { useServerHealth, type ServerStatus } from '../lib/health'
import { useI18n, type TKey } from '../lib/i18n'
import { useTheme } from '../lib/theme'

const NAV: { to: '/' | '/insights' | '/generate' | '/models'; key: TKey; Icon: typeof Search }[] = [
  { to: '/', key: 'nav.analyze', Icon: Search },
  { to: '/insights', key: 'nav.insights', Icon: BarChart3 },
  { to: '/generate', key: 'nav.generate', Icon: LayoutGrid },
  { to: '/models', key: 'nav.models', Icon: ListChecks },
]

export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="8.5" stroke="currentColor" strokeWidth="2" />
      <path d="M18.5 18.5L25 25" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
      <path d="M7.5 13.5h9" stroke="var(--pos)" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

const STATUS_STYLE: Record<ServerStatus | 'demo', { box: string; dot: string; key: TKey }> = {
  connecting: { box: 'bg-sunken text-ink-2', dot: 'bg-ink-3 animate-pulse', key: 'status.connecting' },
  waking: { box: 'bg-neu-soft text-neu-ink', dot: 'bg-neu animate-pulse', key: 'status.waking' },
  ready: { box: 'bg-pos-soft text-pos-ink', dot: 'bg-pos', key: 'status.ready' },
  offline: { box: 'bg-neg-soft text-neg-ink', dot: 'bg-neg', key: 'status.offline' },
  missing: { box: 'bg-neg-soft text-neg-ink', dot: 'bg-neg', key: 'status.missing' },
  demo: { box: 'bg-neu-soft text-neu-ink', dot: 'bg-neu', key: 'status.demo' },
}

export function StatusChip({ compact = false }: { compact?: boolean }) {
  const { t } = useI18n()
  const { status, health } = useServerHealth()
  const s = STATUS_STYLE[health?.mode === 'stub' && status === 'ready' ? 'demo' : status]
  if (compact) {
    return <span role="status" aria-label={t(s.key)} title={t(s.key)} className={`size-2.5 rounded-full ${s.dot}`} />
  }
  return (
    <span role="status" className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[13px] font-medium ${s.box}`}>
      <span aria-hidden className={`size-2 rounded-full ${s.dot}`} />
      {t(s.key)}
    </span>
  )
}

function LanguageToggle() {
  const { lang, setLang, t } = useI18n()
  return (
    <div role="group" aria-label={t('ui.language')} className="inline-flex overflow-hidden rounded-[10px] border border-line-strong">
      {(['vi', 'en'] as const).map((l) => (
        <button
          key={l}
          type="button"
          aria-pressed={lang === l}
          onClick={() => setLang(l)}
          className={`min-h-9 min-w-11 cursor-pointer text-[13px] font-semibold uppercase ${lang === l ? 'bg-primary text-on-primary' : 'bg-surface text-ink-2 hover:bg-sunken'}`}
        >
          {l}
        </button>
      ))}
    </div>
  )
}

function ThemeToggle() {
  const { t } = useI18n()
  const [theme, toggle] = useTheme()
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === 'dark' ? t('ui.toLight') : t('ui.toDark')}
      className="inline-flex size-10 cursor-pointer items-center justify-center rounded-[10px] border border-line-strong bg-surface text-ink hover:bg-sunken"
    >
      {theme === 'dark' ? <Sun aria-hidden className="size-[18px]" /> : <Moon aria-hidden className="size-[18px]" />}
    </button>
  )
}

function DemoBanner() {
  const { t } = useI18n()
  const { health } = useServerHealth()
  if (health?.mode !== 'stub') return null
  return <p className="m-0 bg-neu-soft px-6 py-2 text-center text-[13px] text-neu-ink">{t('demo.banner')}</p>
}

export function Layout({ children }: { children?: ReactNode }) {
  const { t } = useI18n()
  return (
    <div className="min-h-dvh bg-paper pb-20 text-ink md:pb-0">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2">
        {t('ui.skip')}
      </a>
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex min-h-14 max-w-[1200px] flex-wrap items-center gap-x-7 gap-y-2 px-4 md:min-h-16 md:px-6">
          <Link to="/" className="flex items-center gap-2.5 text-ink no-underline">
            <Logo />
            <span className="text-lg font-bold tracking-tight">ReviewLens</span>
          </Link>
          <nav aria-label={t('nav.main')} className="hidden gap-1 md:flex">
            {NAV.map(({ to, key }) => (
              <Link
                key={to}
                to={to}
                activeOptions={{ exact: to === '/' }}
                className="border-b-2 border-transparent px-3 pb-[18px] pt-5 text-ink-2 no-underline hover:text-ink"
                activeProps={{ className: '!border-ink !text-ink font-semibold', 'aria-current': 'page' }}
              >
                {t(key)}
              </Link>
            ))}
          </nav>
          <div className="flex-1" />
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline-flex">
              <StatusChip />
            </span>
            <span className="sm:hidden">
              <StatusChip compact />
            </span>
            <LanguageToggle />
            <ThemeToggle />
          </div>
        </div>
      </header>
      <DemoBanner />
      <main id="main" className="mx-auto max-w-[1200px] px-4 pb-16 pt-6 md:px-6 md:pt-8">
        {children ?? <Outlet />}
      </main>
      <footer className="mx-auto flex max-w-[1200px] flex-wrap justify-between gap-2 border-t border-line px-4 py-6 text-[13px] text-ink-2 md:px-6">
        <span>{t('footer.text')}</span>
        <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="text-ink-2 hover:text-ink">
          {t('footer.api')}
        </a>
      </footer>
      <nav aria-label={t('nav.main')} className="fixed inset-x-0 bottom-0 z-40 grid h-[72px] grid-cols-4 border-t border-line bg-surface md:hidden">
        {NAV.map(({ to, key, Icon }) => (
          <Link
            key={to}
            to={to}
            activeOptions={{ exact: to === '/' }}
            className="flex flex-col items-center justify-center gap-1 text-xs text-ink-2 no-underline"
            activeProps={{ className: '!text-ink font-semibold', 'aria-current': 'page' }}
          >
            <Icon aria-hidden className="size-[22px]" />
            {t(key)}
          </Link>
        ))}
      </nav>
    </div>
  )
}
