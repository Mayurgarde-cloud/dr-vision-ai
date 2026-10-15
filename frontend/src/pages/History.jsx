import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowDown, ArrowUp, Download, FolderOpen, Search, Trash2 } from 'lucide-react'
import { api, assetUrl } from '../api'
import { useFetch } from '../hooks/useFetch'
import { ClassChip, ErrorBox, PageHeader, Spinner } from '../components/ui'
import { classOf } from '../lib/classes'
import { fmtDateTime } from '../lib/format'

export default function History() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [sort, setSort] = useState({ key: 'id', dir: 'desc' })
  const [actionError, setActionError] = useState(null)

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query), 250)
    return () => clearTimeout(t)
  }, [query])

  const { data, error, loading, reload } = useFetch(() => api.predictions(debounced), [debounced])

  const rows = useMemo(() => {
    const list = [...(data || [])]
    const { key, dir } = sort
    list.sort((a, b) => {
      const av = key === 'created_at' ? a.created_at : a[key]
      const bv = key === 'created_at' ? b.created_at : b[key]
      return (av > bv ? 1 : av < bv ? -1 : 0) * (dir === 'asc' ? 1 : -1)
    })
    return list
  }, [data, sort])

  const toggleSort = (key) => setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' }))

  const remove = async (e, id) => {
    e.stopPropagation()
    if (!window.confirm(`Delete prediction #${id}? This cannot be undone.`)) return
    try { await api.deletePrediction(id); setActionError(null); reload() } catch (err) { setActionError(err.message) }
  }

  const clearAll = async () => {
    if (!window.confirm('Delete ALL prediction history, including stored images? This cannot be undone.')) return
    try { await api.clearHistory(); setActionError(null); reload() } catch (err) { setActionError(err.message) }
  }

  const SortHead = ({ k, children }) => (
    <th className="th">
      <button className="inline-flex items-center gap-1 hover:text-ink" onClick={() => toggleSort(k)}>
        {children}
        {sort.key === k && (sort.dir === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} className="text-brand-500" />)}
      </button>
    </th>
  )

  return (
    <div className="fade-up">
      <PageHeader title="Prediction History" subtitle="Your recent predictions (stored locally)." />
      {actionError && <div className="mb-4"><ErrorBox message={actionError} /></div>}

      <div className="grid gap-5 lg:grid-cols-[1.9fr_1fr]">
        <div>
          <label className="card flex items-center gap-3 px-4 py-3 focus-within:ring-2 focus-within:ring-brand-400">
            <Search size={17} className="text-brand-500" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by class or date…"
              aria-label="Search history"
              className="w-full bg-transparent text-sm text-ink outline-none placeholder:text-muted"
            />
          </label>

          <div className="card mt-4 overflow-hidden">
            {loading && !data ? (
              <Spinner />
            ) : error ? (
              <div className="p-4"><ErrorBox message={error} onRetry={reload} /></div>
            ) : rows.length === 0 ? (
              <div className="px-6 py-16 text-center text-sm text-muted">
                {debounced ? 'No predictions match your search.' : <>No predictions yet. <Link to="/predict" className="font-semibold text-brand-500 hover:underline">Analyse an image</Link></>}
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-left">
                  <thead className="bg-card2">
                    <tr>
                      <SortHead k="id">ID</SortHead>
                      <th className="th">Image</th>
                      <th className="th">Predicted Class</th>
                      <SortHead k="confidence">Top class score</SortHead>
                      <SortHead k="created_at">Date &amp; Time</SortHead>
                      <th className="th text-center">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.id} onClick={() => navigate(`/results/${r.id}`)} className="cursor-pointer border-t border-line hover:bg-soft">
                        <td className="td text-muted">#{String(r.id).padStart(2, '0')}</td>
                        <td className="td">
                          <img src={assetUrl(r.image_url)} alt="" loading="lazy" className="h-10 w-10 rounded-lg bg-black object-cover" />
                        </td>
                        <td className="td">
                          <ClassChip index={r.class_index} />
                          <div className="mt-0.5 text-[10.5px] font-medium" style={{ color: classOf(r.class_index).color }}>Class {r.class_index}</div>
                        </td>
                        <td className="td tabular-nums">{r.confidence.toFixed(2)}%</td>
                        <td className="td whitespace-nowrap text-xs text-muted">{fmtDateTime(r.created_at)}</td>
                        <td className="td">
                          <div className="flex items-center justify-center gap-2">
                            <a
                              href={api.reportUrl(r.id)}
                              download
                              onClick={(e) => e.stopPropagation()}
                              aria-label={`Download report for prediction ${r.id}`}
                              className="grid h-8 w-8 place-items-center rounded-lg border border-line text-brand-500 hover:bg-card2"
                            >
                              <Download size={15} />
                            </a>
                            <button
                              onClick={(e) => remove(e, r.id)}
                              aria-label={`Delete prediction ${r.id}`}
                              className="grid h-8 w-8 place-items-center rounded-lg border border-red-200 text-red-500 hover:bg-red-50 dark:border-red-500/30 dark:hover:bg-red-500/10"
                            >
                              <Trash2 size={15} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        <aside className="card flex flex-col items-center px-6 py-8 text-center">
          <span className="grid h-16 w-16 place-items-center rounded-2xl bg-brand-500 text-white shadow-btn"><FolderOpen size={30} /></span>
          <h2 className="mt-5 text-xl font-bold text-ink">Storage Information</h2>
          <p className="mt-3 text-[13px] leading-relaxed text-muted">
            Images and prediction records are stored locally in a SQLite database on the machine running this application.
          </p>
          <p className="mt-3 text-[13px] text-muted">You can clear the history anytime.</p>
          <button onClick={clearAll} disabled={!data?.length} className="btn btn-outline mt-6 px-8">Clear All History</button>
        </aside>
      </div>
    </div>
  )
}
