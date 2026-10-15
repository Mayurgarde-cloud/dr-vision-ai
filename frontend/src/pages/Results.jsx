import { Link, useParams } from 'react-router-dom'
import { Download, FileSearch, Info, TriangleAlert } from 'lucide-react'
import { api, assetUrl } from '../api'
import { useFetch } from '../hooks/useFetch'
import { useTheme } from '../hooks/useTheme'
import { ErrorBox, Disclaimer, Spinner } from '../components/ui'
import { classOf } from '../lib/classes'
import { fmtDateTime } from '../lib/format'

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg bg-card2 px-3 py-2.5 text-[13px]">
      <span className="font-semibold text-brand-700 dark:text-brand-300">{label}</span>
      <span className="text-right text-ink">{value}</span>
    </div>
  )
}

function ConfidenceBars({ probabilities, predicted }) {
  return (
    <ul className="space-y-3.5">
      {probabilities.map((p) => {
        const c = classOf(p.index)
        const top = p.index === predicted
        return (
          <li key={p.index} className="grid grid-cols-[112px_1fr_52px] items-center gap-3 text-[12.5px]">
            <span className={top ? 'font-semibold' : 'text-muted'} style={top ? { color: c.color } : undefined}>
              {p.name} ({p.index})
            </span>
            <span className="h-3 overflow-hidden rounded-full bg-soft" role="img" aria-label={`${p.name}: ${p.probability.toFixed(2)} percent`}>
              <span
                className="block h-full rounded-full transition-all duration-700"
                style={{
                  width: `${Math.max(p.probability, 1.5)}%`,
                  background: top ? 'linear-gradient(90deg,#19c3ff,#0b7ff0)' : c.color,
                  opacity: top ? 1 : 0.7,
                }}
              />
            </span>
            <span className={`text-right tabular-nums ${top ? 'font-bold text-ink' : 'text-muted'}`}>{p.probability.toFixed(2)}%</span>
          </li>
        )
      })}
    </ul>
  )
}

export default function Results() {
  const { id } = useParams()
  const { dark } = useTheme()
  const { data: r, error, loading, reload } = useFetch(() => api.prediction(id), [id])

  if (loading) return <Spinner label="Loading result…" />
  if (error) {
    return (
      <div className="space-y-4">
        <ErrorBox message={error} onRetry={reload} />
        <Link to="/predict" className="btn btn-primary">Analyse an image</Link>
      </div>
    )
  }

  const c = classOf(r.class_index)
  const demo = r.model.mode === 'demo'
  const gradTitle = 'Grad-CAM'

  return (
    <div className="fade-up">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-2">
        <h1 className="page-title">Prediction Results</h1>
        <span className="text-[11.5px] text-muted">Predicted on: {fmtDateTime(r.created_at)} · #{r.id}</span>
      </div>

      {r.warnings?.map((w) => (
        <div key={w} role="alert" className="mb-4 flex gap-2.5 rounded-xl border border-amber-300/70 bg-amber-50 px-4 py-3 text-[13px] text-amber-900 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
          <TriangleAlert size={17} className="mt-0.5 shrink-0" /> {w}
        </div>
      ))}

      <div className="grid gap-5 md:grid-cols-[1fr_1fr] lg:grid-cols-[1.1fr_1fr_1.25fr]">
        <section className="card p-4">
          <h2 className="panel-title mb-3">Original Image</h2>
          <img src={assetUrl(r.image_url)} alt="Uploaded retinal fundus" className="aspect-square w-full rounded-xl bg-black object-contain" />
        </section>

        <section className="card flex flex-col p-5">
          <h2 className="panel-title mb-4">Predicted Severity</h2>
          <div
            className="rounded-xl px-4 py-3.5 text-center text-xl font-extrabold text-white shadow-lg"
            style={{ background: `linear-gradient(180deg, ${c.color}, ${c.color}cc)`, boxShadow: `0 10px 24px -10px ${c.color}` }}
          >
            {r.class_name}
          </div>
          <div className="mx-auto mt-3 rounded-lg px-6 py-1.5 text-[13px] font-semibold" style={{ background: dark ? c.darkSoft : c.soft, color: dark ? c.color : c.text }}>
            Class {r.class_index}
          </div>
          <div className="mt-9">
            <div className="text-sm text-muted">Top class score</div>
            <div className="mt-1 text-4xl font-extrabold tabular-nums text-brand-500">{r.confidence.toFixed(2)}%</div>
          </div>
        </section>

        <section className="card p-5 md:col-span-2 lg:col-span-1">
          <h2 className="panel-title mb-4">Model Information</h2>
          <div className="space-y-2">
            <Row label="Model" value={r.model.backbones} />
            <Row label="Classifier" value={r.model.classifier} />
            <Row label="Image Size" value={r.model.image_size} />
            <Row label="Processing Time" value={`${r.processing_time.toFixed(2)} seconds`} />
            <Row label="Mode" value={demo ? 'Demo (heuristic)' : 'Trained model'} />
            <Row label="Version" value={r.model.version} />
          </div>
        </section>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1.05fr_1fr]">
        <section className="card p-5">
          <h2 className="panel-title mb-2">Class scores</h2>
          <p className="mb-4 text-xs text-muted">{r.model.score_description}</p>
          <ConfidenceBars probabilities={r.probabilities} predicted={r.class_index} />
        </section>

        <section className="card p-5">
          <h2 className="panel-title">{gradTitle}</h2>
          <p className="mb-3 text-[11.5px] text-muted">
            {demo ? 'Regions with lesion-like features found by the demo heuristic.' : 'Regions that most influenced the predicted class.'}
          </p>
          <div className="grid grid-cols-3 gap-2.5 text-center text-[11.5px] text-muted">
            {[['Original', r.image_url], ['Heatmap', r.heatmap_url], ['Overlay', r.overlay_url]].map(([label, url]) => (
              <figure key={label}>
                <img src={assetUrl(url)} alt={`${label} view`} className="aspect-square w-full rounded-lg bg-black object-cover" />
                <figcaption className="mt-1.5">{label}</figcaption>
              </figure>
            ))}
          </div>
        </section>
      </div>

      <section className="card mt-5 flex items-start gap-4 p-5">
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-glow/90 text-white">
          <FileSearch size={21} />
        </span>
        <div>
          <h2 className="panel-title">About this Prediction</h2>
          <p className="mt-2 text-[13.5px] leading-relaxed text-muted">
            The model has predicted <strong className="text-ink">{r.class_name === 'No DR' ? 'No Diabetic Retinopathy' : `${r.class_name.replace(' DR', '')} Diabetic Retinopathy`}</strong>{' '}
            (Class {r.class_index}) with a top class score of {r.confidence.toFixed(2)}%. This indicates {r.explanation}.
          </p>
          {demo && (
            <p className="mt-2 flex items-start gap-2 text-[12.5px] text-amber-700 dark:text-amber-300">
              <Info size={15} className="mt-0.5 shrink-0" /> {r.model.mode_note}
            </p>
          )}
        </div>
      </section>

      <div className="mt-5 flex flex-wrap gap-4">
        <a href={api.reportUrl(r.id)} className="btn btn-primary px-8" download>
          Download Report <Download size={16} />
        </a>
        <Link to="/predict" className="btn btn-outline px-8">Analyse Another Image</Link>
      </div>

      <div className="mt-5">
        <Disclaimer>This prediction is for educational and research purposes only. It is not a substitute for professional medical advice.</Disclaimer>
      </div>
    </div>
  )
}
