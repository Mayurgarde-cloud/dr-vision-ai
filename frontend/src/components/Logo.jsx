import { Link } from 'react-router-dom'

export function EyeMark({ className = 'h-9 w-12' }) {
  return (
    <svg viewBox="0 0 64 44" className={className} fill="none" aria-hidden="true">
      <path d="M3 22C12 8 22 3 32 3s20 5 29 19c-9 14-19 19-29 19S12 36 3 22Z" stroke="#19c3ff" strokeWidth="3.2" strokeLinejoin="round" />
      <circle cx="32" cy="22" r="11" stroke="#19c3ff" strokeWidth="3" />
      <circle cx="32" cy="22" r="5" fill="#19c3ff" />
      <circle cx="35" cy="19" r="1.6" fill="#061a3f" />
    </svg>
  )
}

export default function Logo({ to = '/', compact = false }) {
  return (
    <Link to={to} className="flex items-center gap-3" aria-label="DR Vision AI home">
      <EyeMark />
      <span className="leading-tight">
        <span className="block text-[19px] font-bold tracking-tight text-white">DR Vision AI</span>
        {!compact && <span className="block text-[11px] font-medium text-glow">AI for Better Eye Care</span>}
      </span>
    </Link>
  )
}
