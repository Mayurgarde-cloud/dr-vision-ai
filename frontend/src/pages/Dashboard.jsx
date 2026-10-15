import { Link, useNavigate } from 'react-router-dom'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { BarChart3, Database, FileClock, UploadCloud } from 'lucide-react'
import { api } from '../api'
import { useFetch } from '../hooks/useFetch'
import { ClassChip, ErrorBox, PageHeader, Spinner } from '../components/ui'
import { CLASSES, classOf } from '../lib/classes'
import { fmtDate, fmtDateTime, fmtTime } from '../lib/format'

function Stat({ label, value, valueClass = 'text-ink', sub }) {
  return (
    <div className="card p-4">
      <div className="text-[13px] font-semibold text-brand-700 dark:text-brand-300">{label}</div>
      <div className={`mt-2 text-[22px] font-extrabold leading-tight ${valueClass}`}>{value}</div>
      <div className="mt-1 text-[11.5px] text-muted">{sub}</div>
    </div>
  )
}

const ACTIONS = [
  { to: '/predict', icon: UploadCloud, title: 'Upload Image', sub: 'Make a new prediction' },
  { to: '/history', icon: FileClock, title: 'View History', sub: 'Check past results' },
  { to: '/performance', icon: BarChart3, title: 'Model Performance', sub: 'See evaluation metrics' },
  { to: '/dataset', icon: Database, title: 'Dataset Info', sub: 'View dataset details' },
]

export default function Dashboard() {
  const navigate = useNavigate()
  const { data: d, error, loading, reload } = useFetch(api.dashboard, [])

  if (loading) return <Spinner />
  if (error) return <ErrorBox message={error} onRetry={reload} />

  const loaded = d.dataset.status === 'Loaded'
  const ready = d.model.mode === 'trained'
  const total = d.dataset.total
  const chartData = d.class_distribution.map((c) => ({ ...c, fill: classOf(c.index).color }))

  return (
    <div className="fade-up">
      <PageHeader title="Project Dashboard" subtitle="Overview of model performance, dataset and recent predictions." />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat
          label="Dataset Status"
          value={d.dataset.status}
          valueClass={loaded ? 'text-emerald-500' : 'text-amber-500'}
          sub={total ? `${total} images` : 'No images found'}
        />
        <Stat
          label="Model Status"
          value={d.model.status}
          valueClass={ready ? 'text-emerald-500' : 'text-amber-500'}
          sub={d.model.detail}
        />
        <Stat
          label="Last Prediction"
          value={d.last_prediction ? fmtDate(d.last_prediction) : '—'}
          sub={d.last_prediction ? fmtTime(d.last_prediction) : 'No predictions yet'}
        />
        <Stat label="Total Predictions" value={d.total_predictions} sub="(All time)" />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_1.15fr]">
        <section className="card p-5">
          <h2 className="panel-title mb-3">Class Distribution</h2>
          {total === 0 ? (
            <p className="py-16 text-center text-sm text-muted">No dataset loaded yet.</p>
          ) : (
            <div className="flex flex-col items-center gap-5 sm:flex-row">
              <div className="relative h-[190px] w-[190px] shrink-0">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={chartData} dataKey="count" nameKey="name" innerRadius={58} outerRadius={92} paddingAngle={1.5} stroke="none">
                      {chartData.map((e) => <Cell key={e.index} fill={e.fill} />)}
                    </Pie>
                    <Tooltip formatter={(v, n) => [`${v} images`, n]} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
                  <div>
                    <div className="text-2xl font-extrabold text-ink">{total}</div>
                    <div className="text-[11px] text-muted">images</div>
                  </div>
                </div>
              </div>
              <ul className="w-full space-y-2.5 text-[13px]">
                {d.class_distribution.map((c) => (
                  <li key={c.index} className="flex items-center gap-2.5">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: classOf(c.index).color }} />
                    <span className="flex-1 text-muted">{c.name}</span>
                    <span className="font-semibold tabular-nums text-ink">{Math.round(c.percent)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        <section className="card overflow-hidden p-5">
          <h2 className="panel-title mb-3">Recent Predictions</h2>
          {d.recent.length === 0 ? (
            <div className="py-12 text-center text-sm text-muted">
              No predictions yet. <Link to="/predict" className="font-semibold text-brand-500 hover:underline">Analyse an image</Link>
            </div>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-line">
              <table className="w-full min-w-[420px] text-left">
                <thead className="bg-card2">
                  <tr><th className="th">ID</th><th className="th">Class</th><th className="th">Top class score</th><th className="th">Time</th></tr>
                </thead>
                <tbody>
                  {d.recent.map((r) => (
                    <tr
                      key={r.id}
                      onClick={() => navigate(`/results/${r.id}`)}
                      className="cursor-pointer border-t border-line hover:bg-soft"
                    >
                      <td className="td text-muted">#{r.id}</td>
                      <td className="td"><ClassChip index={r.class_index} /></td>
                      <td className="td tabular-nums">{r.confidence.toFixed(2)}%</td>
                      <td className="td whitespace-nowrap text-xs text-muted">{fmtDateTime(r.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      <section className="mt-6">
        <h2 className="panel-title mb-3">Quick Actions</h2>
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {ACTIONS.map(({ to, icon: Icon, title, sub }) => (
            <Link key={to} to={to} className="card flex flex-col items-center gap-2 px-3 py-5 text-center transition hover:-translate-y-0.5 hover:border-brand-400">
              <span className="grid h-12 w-12 place-items-center rounded-full bg-soft text-brand-500"><Icon size={22} /></span>
              <span className="text-[13.5px] font-bold text-ink">{title}</span>
              <span className="text-[11.5px] text-muted">{sub}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}
