// The five DR severity classes and the colour used for each across the whole app.
export const CLASSES = [
  { index: 0, name: 'No DR', short: 'Healthy retina', color: '#16a34a', soft: '#e7f8ed', darkSoft: '#0f3a26', text: '#15803d' },
  { index: 1, name: 'Mild DR', short: 'Early changes', color: '#f5b301', soft: '#fff4d3', darkSoft: '#4a3a0a', text: '#a16207' },
  { index: 2, name: 'Moderate DR', short: 'More visible signs', color: '#f97316', soft: '#ffebdb', darkSoft: '#4d2410', text: '#c2570c' },
  { index: 3, name: 'Severe DR', short: 'Advanced damage', color: '#ef4444', soft: '#ffe5e5', darkSoft: '#4d1717', text: '#b91c1c' },
  { index: 4, name: 'Proliferative DR', short: 'High risk of vision loss', color: '#8b5cf6', soft: '#eee8ff', darkSoft: '#2e1f5c', text: '#6d3fd6' },
]

export const classOf = (i) => CLASSES[Math.max(0, Math.min(4, Number(i) || 0))]
