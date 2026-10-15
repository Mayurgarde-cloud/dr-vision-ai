import { Link, useLocation } from 'react-router-dom'
import { NAV, isActive } from '../lib/nav'

export default function Sidebar() {
  const { pathname } = useLocation()
  return (
    <aside
  className="hidden w-[236px] shrink-0 flex-col gap-1.5 bg-navy-900 px-4 pb-6 pt-5 lg:flex"
  aria-label="Sidebar"
>
      {NAV.map((item) => {
        const active = isActive(item, pathname)
        const label = item.to === '/predict' && pathname.startsWith('/results') ? 'Results' : item.label
        return (
          <Link
            key={item.to}
            to={item.to}
            aria-current={active ? 'page' : undefined}
            className={`flex items-center gap-3.5 rounded-xl px-4 py-3 text-[13.5px] font-medium transition ${
              active
                ? 'bg-gradient-to-r from-brand-500 to-brand-700 text-white shadow-glow'
                : 'text-white/85 hover:bg-white/10 hover:text-white'
            }`}
          >
            <item.icon size={19} strokeWidth={1.9} />
            {label}
          </Link>
        )
      })}
    </aside>
  )
}
