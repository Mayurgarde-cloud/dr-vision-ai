import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Menu, Moon, Sun, X } from 'lucide-react'
import Logo from './Logo'
import { NAV, isActive } from '../lib/nav'
import { useTheme } from '../hooks/useTheme'

function ThemeSwitch() {
  const { dark, toggle } = useTheme()
  return (
    <button
      onClick={toggle}
      role="switch"
      aria-checked={dark}
      aria-label="Toggle dark mode"
      className="flex h-7 w-[58px] items-center gap-1 rounded-full border border-white/10 bg-white/10 px-1.5 text-white/70 transition hover:bg-white/15"
    >
      <Moon size={13} className={dark ? 'text-glow' : 'opacity-70'} />
      <span className={`h-5 w-5 rounded-full bg-white shadow transition-transform ${dark ? 'translate-x-0' : 'translate-x-[18px]'}`} />
      <Sun size={0} />
    </button>
  )
}

export default function TopNav({ className = '' }) {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)

  return (
    <header className={`relative z-30 ${className}`}>
      <div className="flex h-[68px] items-center justify-between gap-4 px-4 sm:px-8">
        <Logo />
        <nav className="hidden items-center gap-7 lg:flex" aria-label="Main">
          {NAV.map((item) => {
            const active = isActive(item, pathname)
            return (
              <Link
                key={item.to}
                to={item.to}
                aria-current={active ? 'page' : undefined}
                className={`relative py-1 text-[13px] font-medium transition-colors ${active ? 'text-glow' : 'text-white/90 hover:text-white'}`}
              >
                {item.label}
                {active && <span className="absolute -bottom-1 left-0 h-[2px] w-full rounded bg-glow" />}
              </Link>
            )
          })}
        </nav>
        <div className="flex items-center gap-3">
          <ThemeSwitch />
          <button
            className="rounded-lg p-2 text-white/90 hover:bg-white/10 lg:hidden"
            onClick={() => setOpen((o) => !o)}
            aria-label={open ? 'Close menu' : 'Open menu'}
            aria-expanded={open}
          >
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>

      {open && (
        <nav className="absolute inset-x-0 top-full border-t border-white/10 bg-navy-900/98 px-4 pb-4 pt-2 shadow-2xl backdrop-blur lg:hidden" aria-label="Mobile">
          {NAV.map((item) => {
            const active = isActive(item, pathname)
            return (
              <Link
                key={item.to}
                to={item.to}
                onClick={() => setOpen(false)}
                className={`flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium ${active ? 'bg-brand-600 text-white' : 'text-white/85 hover:bg-white/10'}`}
              >
                <item.icon size={18} /> {item.label}
              </Link>
            )
          })}
        </nav>
      )}
    </header>
  )
}
