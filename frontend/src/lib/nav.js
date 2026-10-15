import { Database, Gauge, History, Home, Info, LayoutDashboard, ScanEye } from 'lucide-react'

export const NAV = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/predict', label: 'Predict', icon: ScanEye },
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/dataset', label: 'Dataset', icon: Database },
  { to: '/performance', label: 'Model Performance', icon: Gauge },
  { to: '/history', label: 'History', icon: History },
  { to: '/about', label: 'About', icon: Info },
]

export function isActive(item, pathname) {
  if (item.to === '/') return pathname === '/'
  if (item.to === '/predict') return pathname.startsWith('/predict') || pathname.startsWith('/results')
  return pathname.startsWith(item.to)
}
