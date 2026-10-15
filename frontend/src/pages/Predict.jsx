import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CheckCircle2, ImageIcon, Loader2, UploadCloud, X } from 'lucide-react'
import { api } from '../api'
import { PageHeader, ErrorBox } from '../components/ui'
import { CLASSES } from '../lib/classes'
import { fmtBytes } from '../lib/format'

const MAX_BYTES = 5 * 1024 * 1024
const OK_TYPES = ['image/jpeg', 'image/png']

function validate(file) {
  const ext = file.name.split('.').pop()?.toLowerCase()
  if (!OK_TYPES.includes(file.type) || !['jpg', 'jpeg', 'png'].includes(ext)) {
    return 'Unsupported file type. Please choose a JPG, JPEG or PNG image.'
  }
  if (file.size > MAX_BYTES) return `File is too large (${fmtBytes(file.size)}). The maximum size is 5 MB.`
  if (file.size === 0) return 'The selected file is empty.'
  return null
}

const REQUIREMENTS = ['JPG / JPEG / PNG format', 'Max file size: 5MB', 'Clear, uncorrupted image', 'Retinal fundus image']

export default function Predict() {
  const navigate = useNavigate()
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!file) { setPreview(null); return undefined }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const choose = useCallback((f) => {
    if (!f) return
    const problem = validate(f)
    if (problem) { setError(problem); setFile(null); return }
    setError(null)
    setFile(f)
  }, [])

  const clear = () => {
    setFile(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const analyse = async () => {
    if (!file || busy) return
    setBusy(true)
    setError(null)
    try {
      const result = await api.predict(file)
      navigate(`/results/${result.id}`)
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  const loadSample = async (index) => {
    try {
      const blob = await (await fetch(`/samples/class_${index}.jpg`)).blob()
      choose(new File([blob], `sample_${CLASSES[index].name.replace(/\s+/g, '_').toLowerCase()}.jpg`, { type: 'image/jpeg' }))
    } catch {
      setError('Could not load the sample image.')
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    choose(e.dataTransfer.files?.[0])
  }

  return (
    <div className="fade-up">
      <PageHeader
        title="Predict Diabetic Retinopathy"
        subtitle="Upload a retinal fundus image to view the research model's predicted severity and class scores."
      />

      {/* drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`rounded-2xl border-2 border-dashed px-6 py-10 text-center transition ${
          dragging ? 'border-brand-500 bg-soft' : 'border-brand-300/70 bg-gradient-to-b from-card2 to-card'
        }`}
      >
        <UploadCloud size={54} strokeWidth={1.6} className="mx-auto text-brand-500" />
        <p className="mt-3 text-[15px] font-bold text-brand-700 dark:text-brand-300">Drag and drop an image here</p>
        <p className="my-1.5 text-sm font-semibold text-muted">or</p>
        <button type="button" className="btn btn-primary px-8" onClick={() => inputRef.current?.click()}>
          Browse Files
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          className="hidden"
          onChange={(e) => choose(e.target.files?.[0])}
          aria-label="Choose a retinal fundus image"
        />
        <p className="mt-4 text-xs text-muted">Supported formats: JPG, JPEG, PNG &nbsp;|&nbsp; Max size: 5MB</p>
      </div>

      {error && <div className="mt-4"><ErrorBox message={error} /></div>}

      <div className="mt-5 grid gap-5 md:grid-cols-[1.35fr_1fr]">
        {/* preview */}
        <section className="card p-4">
          <h2 className="panel-title mb-3">Preview</h2>
          <div className="relative grid min-h-[230px] place-items-center overflow-hidden rounded-xl border border-line bg-gradient-to-b from-card2 to-soft">
            {preview ? (
              <>
                <img src={preview} alt="Selected retinal fundus" className="max-h-[300px] w-full object-contain" />
                <button onClick={clear} aria-label="Remove image" className="absolute right-2 top-2 rounded-full bg-black/60 p-1.5 text-white hover:bg-black/80">
                  <X size={14} />
                </button>
              </>
            ) : (
              <div className="flex flex-col items-center gap-3 py-8 text-sm text-muted">
                <span className="grid h-20 w-20 place-items-center rounded-full bg-soft text-brand-400">
                  <ImageIcon size={36} strokeWidth={1.5} />
                </span>
                No image uploaded yet
              </div>
            )}
          </div>
          {file && <p className="mt-2 truncate text-xs text-muted">{file.name} · {fmtBytes(file.size)}</p>}
        </section>

        {/* requirements */}
        <section className="card p-5">
          <h2 className="panel-title mb-4">Image Requirements</h2>
          <ul className="space-y-3.5">
            {REQUIREMENTS.map((r) => (
              <li key={r} className="flex items-center gap-3 text-sm text-muted">
                <CheckCircle2 size={19} className="shrink-0 text-emerald-500" /> {r}
              </li>
            ))}
          </ul>
        </section>
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-[1.35fr_1fr]">
        <button className="btn btn-primary py-3.5" onClick={analyse} disabled={!file || busy}>
          {busy ? <><Loader2 size={17} className="animate-spin" /> Analysing…</> : 'Analyze Image'}
        </button>
        <button className="btn btn-outline py-3.5" onClick={clear} disabled={busy || (!file && !error)}>Clear</button>
      </div>

      {/* samples */}
      <section className="mt-8">
        <h2 className="text-sm font-semibold text-ink">No fundus image handy? Try a sample</h2>
        <p className="mt-0.5 text-xs text-muted">These are synthetic illustrations (not patient photographs) – click one to load it.</p>
        <div className="mt-3 flex flex-wrap gap-3">
          {CLASSES.map((c) => (
            <button
              key={c.index}
              onClick={() => loadSample(c.index)}
              disabled={busy}
              className="group w-[96px] rounded-xl border border-line bg-card p-1.5 text-center transition hover:border-brand-400 hover:shadow-card disabled:opacity-50"
              aria-label={`Load sample: ${c.name}`}
            >
              <img src={`/samples/class_${c.index}.jpg`} alt="" className="aspect-square w-full rounded-lg bg-black object-cover" />
              <span className="mt-1 block truncate text-[10.5px] font-medium text-muted group-hover:text-ink">{c.name}</span>
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
