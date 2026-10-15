import { TriangleAlert } from 'lucide-react'
import { api } from '../api'
import { useFetch } from '../hooks/useFetch'

/** Shown on every app page while the backend is in demo mode (or running a smoke-test model). */
export default function DemoBanner() {
  const { data } = useFetch(api.health, [])
  if (!data) return null
  const demo = data.model_mode === 'demo'
  if (!demo && !data.note) return null

  const missing = data.reason_code === 'no_model'
  return (
    <div role="status" className="fade-up mb-5 flex gap-3 rounded-xl border border-amber-300/70 bg-amber-50 px-4 py-3 text-[13px] leading-relaxed text-amber-900 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
      <TriangleAlert size={18} className="mt-0.5 shrink-0" />
      <div>
        <strong>{demo ? 'Demo mode.' : 'Model note.'}</strong>{' '}
        {demo && 'Predictions come from a simple image-statistics heuristic and have no diagnostic meaning. '}
        {missing ? (
          <>Train the real ResNet-50 + DenseNet-121 pipeline with <code className="rounded bg-black/5 px-1 dark:bg-white/10">python -m ml.train</code>.</>
        ) : (
          data.note
        )}
      </div>
    </div>
  )
}
