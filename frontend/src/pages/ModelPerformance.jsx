import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Info, LayoutGrid } from 'lucide-react'
import { api } from '../api'
import { useFetch } from '../hooks/useFetch'
import { ErrorBox, PageHeader, Spinner } from '../components/ui'
import { classOf } from '../lib/classes'

const f = (v) => `${(v * 100).toFixed(1)}%`

function Metric({ label, value }) {
  return (
    <div className="card p-4">
      <div className="text-[13px] font-semibold text-brand-700 dark:text-brand-300">{label}</div>
      {value == null ? (
        <div className="mt-2 text-xs text-muted">Not evaluated yet</div>
      ) : (
        <div className="mt-1.5 text-2xl font-extrabold tabular-nums text-ink">{f(value)}</div>
      )}
    </div>
  )
}

function ConfusionMatrix({ matrix, classes }) {
  const rowMax = matrix.map((r) => Math.max(1, r.reduce((a, b) => a + b, 0)))
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[320px] border-separate border-spacing-1 text-center text-xs">
        <thead>
          <tr>
            <th className="p-1 text-[10px] font-medium text-muted">True ↓ / Pred →</th>
            {classes.map((c) => <th key={c.index} className="p-1 font-semibold text-muted">{c.index}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="p-1 text-right font-semibold text-muted">{i}</th>
              {row.map((v, j) => {
                const frac = v / rowMax[i]
                const diag = i === j
                const base = diag ? '11,127,240' : '239,68,68'
                return (
                  <td
                    key={j}
                    title={`True ${classes[i].name} → predicted ${classes[j].name}: ${v}`}
                    className="rounded-md py-3 font-semibold tabular-nums"
                    style={{ background: v ? `rgba(${base},${0.12 + frac * 0.78})` : 'var(--card2)', color: v && frac > 0.45 ? '#fff' : 'var(--ink)' }}
                  >
                    {v}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function ModelPerformance() {
  const perf = useFetch(api.performance, [])
  const ds = useFetch(api.dataset, [])
  if (perf.loading) return <Spinner />
  if (perf.error) return <ErrorBox message={perf.error} onRetry={perf.reload} />

  const p = perf.data
  const ev = p.evaluated
  const dist = (ds.data?.classes || []).map((c) => ({ label: String(c.index), name: c.name, count: c.count, fill: classOf(c.index).color }))

  return (
    <div className="fade-up">
      <PageHeader title="Model Performance" subtitle="Evaluation results on the test set." />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Metric label="Accuracy" value={ev ? p.accuracy : null} />
        <Metric label="Macro Precision" value={ev ? p.macro_precision : null} />
        <Metric label="Macro Recall" value={ev ? p.macro_recall : null} />
        <Metric label="Macro F1-Score" value={ev ? p.macro_f1 : null} />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <section className="card p-5">
          <h2 className="panel-title mb-4">Confusion Matrix</h2>
          {ev ? (
            <ConfusionMatrix matrix={p.confusion_matrix} classes={p.classes} />
          ) : (
            <div className="grid min-h-[250px] place-items-center rounded-xl border border-line bg-soft/60 px-6 text-center">
              <div>
                <span className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-card text-brand-400"><LayoutGrid size={26} /></span>
                <p className="mt-3 text-[13px] text-muted">Evaluation not available yet.</p>
                <p className="mt-1 text-[12px] text-muted">Run the evaluation script to see results.</p>
              </div>
            </div>
          )}
        </section>

        <section className="card p-5">
          <h2 className="panel-title mb-4">Per-Class Metrics</h2>
          <div className="overflow-x-auto rounded-xl border border-line">
            <table className="w-full min-w-[340px] text-left">
              <thead className="bg-card2">
                <tr><th className="th">Class</th><th className="th">Precision</th><th className="th">Recall</th><th className="th">F1-Score</th></tr>
              </thead>
              <tbody>
                {p.classes.map((c, i) => {
                  const m = ev ? p.per_class[i] : null
                  return (
                    <tr key={c.index} className="border-t border-line">
                      <td className="td text-[13px] font-medium text-ink">{c.index} - {c.name}</td>
                      <td className="td tabular-nums text-muted">{m ? f(m.precision) : '-'}</td>
                      <td className="td tabular-nums text-muted">{m ? f(m.recall) : '-'}</td>
                      <td className="td tabular-nums text-muted">{m ? f(m.f1) : '-'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_1.1fr]">
        <section className="card p-5">
          <h2 className="panel-title mb-3">Class Distribution</h2>
          <div className="h-[200px]">
            {dist.some((d) => d.count > 0) ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dist} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" vertical={false} />
                  <XAxis dataKey="label" tick={{ fill: 'var(--muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: 'var(--muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <Tooltip cursor={{ fill: 'var(--soft)' }} formatter={(v, _n, item) => [`${v} images`, item.payload.name]} labelFormatter={() => ''} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {dist.map((d) => <Cell key={d.label} fill={d.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="grid h-full place-items-center text-sm text-muted">No dataset loaded yet.</p>
            )}
          </div>
        </section>

        <section className="card flex items-start gap-3 bg-soft/50 p-5 text-[13px] leading-relaxed text-muted">
          <Info size={17} className="mt-0.5 shrink-0 text-brand-500" />
          {ev ? (
            <div className="space-y-2">
              <p>
                Evaluated on <strong className="text-ink">{p.n_samples}</strong> held-out test images
                {p.val_accuracy != null && <> (validation accuracy {f(p.val_accuracy)})</>}.
              </p>
              {p.config && (
                <p>
                  Config: {p.config.pretrained ? 'ImageNet-pretrained backbones' : 'random backbone weights (smoke test)'},
                  PCA {p.config.pca_components} components, {p.config.classifier || 'AdaBoost/Random Forest'}
                  {p.config.C != null && ` (C=${p.config.C}, class weights=${p.config.class_weight || 'none'})`}.
                </p>
              )}
              {p.config && !p.config.pretrained && (
                <p className="font-semibold text-amber-700 dark:text-amber-300">
                  These numbers come from a smoke-test model with random weights and are not meaningful.
                </p>
              )}
              {p.log_loss != null && <p>Score quality: log loss {p.log_loss.toFixed(3)}, multiclass Brier score {p.multiclass_brier_score.toFixed(3)}. These metrics do not represent clinical probabilities.</p>}
              {p.inference_ms_per_image != null && <p>Cached feature inference: {p.inference_ms_per_image.toFixed(2)} ms per image.</p>}
            </div>
          ) : (
            <p>
              Evaluation results will appear here after training and running the model evaluation script:{' '}
              <code className="rounded bg-card px-1">python -m ml.train</code> then <code className="rounded bg-card px-1">python -m ml.evaluate</code>.
            </p>
          )}
        </section>
      </div>
    </div>
  )
}
