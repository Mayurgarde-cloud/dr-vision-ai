// Thin wrapper around the FastAPI backend. In dev, Vite proxies /api and /files to :8000.
const BASE = import.meta.env.VITE_API_URL || ''

async function request(path, options) {
  let res
  try {
    res = await fetch(BASE + path, options)
  } catch {
    throw new Error('Cannot reach the server. Make sure the backend is running on port 8000.')
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') message = body.detail
      else if (Array.isArray(body.detail)) message = body.detail.map((d) => d.msg).join('; ')
      else if (body.message) message = body.message
    } catch { /* body was not JSON */ }
    throw new Error(message)
  }
  return res.json()
}

export const assetUrl = (u) => (u ? BASE + u : u)

export const api = {
  health: () => request('/api/health'),
  dashboard: () => request('/api/dashboard'),
  dataset: () => request('/api/dataset'),
  performance: () => request('/api/model/performance'),
  predict: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return request('/api/predict', { method: 'POST', body: fd })
  },
  predictions: (search = '') => request(`/api/predictions?search=${encodeURIComponent(search)}`),
  prediction: (id) => request(`/api/predictions/${id}`),
  deletePrediction: (id) => request(`/api/predictions/${id}`, { method: 'DELETE' }),
  clearHistory: () => request('/api/predictions', { method: 'DELETE' }),
  reportUrl: (id) => `${BASE}/api/predictions/${id}/report`,
}
