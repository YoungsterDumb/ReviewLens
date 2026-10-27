import { useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import { api } from './api'
import type { HealthResponse } from './types'

export type ServerStatus = 'connecting' | 'waking' | 'ready' | 'offline' | 'missing'

const CORE = ['ate', 'atsc'] as const
let warmupRequested = false

export function serverStatus(health: HealthResponse | undefined, failed: boolean): ServerStatus {
  if (!health) return failed ? 'offline' : 'connecting'
  if (CORE.some((m) => health.models[m]?.state === 'error')) return 'missing'
  return CORE.every((m) => health.models[m]?.state === 'ready') ? 'ready' : 'waking'
}

/** Poll /health quickly while models load, slowly once they are ready; ask the server to warm up once per page load. */
export function useServerHealth() {
  const query = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    retry: 2,
    refetchInterval: (q) => {
      const data = q.state.data
      if (!data) return 5_000
      return Object.values(data.models).every((m) => m.state === 'ready' || m.state === 'error') ? 60_000 : 3_000
    },
  })
  useEffect(() => {
    if (warmupRequested) return
    warmupRequested = true
    api.warmup().catch(() => {
      warmupRequested = false
    })
  }, [])
  return { health: query.data, status: serverStatus(query.data, query.isError), isError: query.isError }
}
