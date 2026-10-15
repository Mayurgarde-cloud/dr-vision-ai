import { AlertCircle, Loader2, RotateCw } from 'lucide-react'
import { classOf } from '../lib/classes'
import { useTheme } from '../hooks/useTheme'

export function ClassChip({ index, children }) {
  const c = classOf(index)
  const { dark } = useTheme()
  return (
    <span className="chip" style={{ background: dark ? c.darkSoft : c.soft, color: dark ? c.color : c.text }}>
      {children ?? c.name}
    </span>
  )
}

export function PageHeader({ title, subtitle, right }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-sub">{subtitle}</p>}
      </div>
      {right}
    </div>
  )
}

export function Spinner({ label = 'Loading…' }) {
  return (
    <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted" role="status">
      <Loader2 className="animate-spin" size={18} /> {label}
    </div>
  )
}

export function ErrorBox({ message, onRetry }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-red-300/70 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
      <AlertCircle size={18} className="mt-0.5 shrink-0" />
      <div className="flex-1">{message}</div>
      {onRetry && (
        <button onClick={onRetry} className="inline-flex items-center gap-1 font-semibold underline-offset-2 hover:underline">
          <RotateCw size={14} /> Retry
        </button>
      )}
    </div>
  )
}

export function Disclaimer({ children }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-line bg-card2 px-4 py-3 text-[12.5px] text-brand-700 dark:text-brand-300">
      <AlertCircle size={15} className="mt-0.5 shrink-0" />
      <span>{children}</span>
    </div>
  )
}
