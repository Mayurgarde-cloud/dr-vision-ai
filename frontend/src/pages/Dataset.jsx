import { CheckCircle2, Circle, Layers3, ShieldQuestion, FlaskConical, Dumbbell } from 'lucide-react'
import { api } from '../api'
import { useFetch } from '../hooks/useFetch'
import { ErrorBox, PageHeader, Spinner } from '../components/ui'
import { classOf } from '../lib/classes'

const SPLIT_ICONS = { Training: Dumbbell, Validation: ShieldQuestion, Test: FlaskConical }

function StatusChip({ status }) {
  const ok = status === 'Loaded'
  const cls = ok
    ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300'
    : 'bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300'
  return <span className={`chip px-3 py-1 text-xs ${cls}`}>{status}</span>
}

export default function Dataset() {
  const { data: d, error, loading, reload } = useFetch(api.dataset, [])
  if (loading) return <Spinner />
  if (error) return <ErrorBox message={error} onRetry={reload} />

  const max = Math.max(1, ...d.classes.map((c) => c.count))

  return (
    <div className="fade-up">
      <PageHeader title="Dataset Information" subtitle="Details about the dataset used for training and evaluation." />

      {d.hint && <div className="mb-5"><ErrorBox message={d.hint} /></div>}

      <section className="card overflow-hidden">
        <div className="border-b border-line bg-soft/60 px-5 py-3 text-sm font-bold text-ink">Dataset Overview</div>
        <div className="grid grid-cols-2 gap-y-4 p-5 sm:grid-cols-4">
          {[
            ['Name', d.name],
            ['Source', d.source],
            ['Total Images', d.total],
            ['Status', <StatusChip key="s" status={d.status} />],
          ].map(([k, v]) => (
            <div key={k} className="sm:border-l sm:border-line sm:pl-5 sm:first:border-0 sm:first:pl-0">
              <div className="text-[13px] font-semibold text-ink">{k}</div>
              <div className="mt-2 break-words text-sm text-muted">{v}</div>
            </div>
          ))}
        </div>

        <div className="border-t border-line px-5 pb-6 pt-4">
          <h2 className="panel-title mb-4">Class Distribution</h2>
          <ul className="space-y-3.5">
            {d.classes.map((c) => {
              const col = classOf(c.index).color
              return (
                <li key={c.index} className="grid grid-cols-[1fr_88px] items-center gap-3 sm:grid-cols-[1fr_96px]">
                  <div className="h-8 overflow-hidden rounded-lg bg-soft">
                    <div
                      className="flex h-full items-center rounded-lg px-3 text-xs font-semibold text-white transition-all duration-700"
                      style={{ width: `${Math.max((c.count / max) * 100, c.count ? 14 : 0)}%`, background: col }}
                    >
                      <span className="truncate">{c.name}</span>
                    </div>
                  </div>
                  <span className="text-right text-[13px] font-semibold tabular-nums text-ink">
                    {c.count} <span className="font-normal text-muted">({Math.round(c.percent)}%)</span>
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      </section>

      <div className="mt-5 grid gap-5 md:grid-cols-[1.25fr_1fr]">
        <section className="card p-5">
          <h2 className="panel-title mb-4">Preprocessing Steps</h2>
          <ul className="space-y-4">
            {d.preprocessing.map((s) => (
              <li key={s.label} className={`flex items-center gap-3 text-sm ${s.active ? 'text-ink' : 'text-muted'}`}>
                {s.active ? <CheckCircle2 size={19} className="text-brand-500" /> : <Circle size={19} className="text-line" />}
                <span>{s.label}{!s.active && <span className="ml-2 text-[11px]">(not used by the current model)</span>}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="card p-5">
          <h2 className="panel-title mb-4">Dataset Split</h2>
          <ul className="space-y-2.5">
            {d.split.map((s) => {
              const Icon = SPLIT_ICONS[s.name] || Layers3
              return (
                <li key={s.name} className="flex items-center gap-3 rounded-xl border border-line bg-card2 px-3.5 py-3 text-[13px]">
                  <Icon size={17} className="text-brand-500" />
                  <span className="flex-1 text-muted">{s.name}</span>
                  <span className="font-semibold tabular-nums text-ink">{s.percent}% ({s.count})</span>
                </li>
              )
            })}
          </ul>
          {!d.split_is_actual && d.total > 0 && (
            <p className="mt-3 text-[11.5px] text-muted">Planned split – the exact counts are recorded when you run <code>python -m ml.train</code>.</p>
          )}
        </section>
      </div>
    </div>
  )
}
