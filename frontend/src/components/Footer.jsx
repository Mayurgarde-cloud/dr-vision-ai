import { Link } from 'react-router-dom'
import { Github, Linkedin, Twitter, Instagram } from 'lucide-react'
import { EyeMark } from './Logo'
import { NAV } from '../lib/nav'

const TECH = [
  { name: 'Python', color: '#f5c518' },
  { name: 'PyTorch', color: '#ee4c2c' },
  { name: 'React', color: '#61dafb' },
  { name: 'FastAPI', color: '#05c8a7' },
  { name: 'scikit-learn', color: '#3b9ddd' },
]

export default function Footer() {
  return (
    <footer className="bg-navy-900 text-white">
      <div className="mx-auto grid max-w-6xl gap-10 px-6 py-12 sm:px-8 md:grid-cols-[1.5fr_1fr_1fr_1fr]">
        <div>
          <div className="flex items-center gap-3">
            <EyeMark className="h-9 w-12" />
            <div className="leading-tight">
              <div className="text-[19px] font-bold">DR Vision AI</div>
              <div className="text-[11px] font-medium text-glow">AI for Better Eye Care</div>
            </div>
          </div>
          <p className="mt-5 max-w-xs text-sm leading-relaxed text-white/75">
            An AI-powered educational application for retinal image analysis and diabetic retinopathy severity classification using deep learning.
          </p>
          <div className="mt-5 flex gap-4 text-white/85">
            {[Github, Linkedin, Instagram, Twitter].map((Icon, i) => (
              <span key={i} className="cursor-default" aria-hidden="true"><Icon size={18} /></span>
            ))}
          </div>
        </div>

        <div>
          <h4 className="mb-4 text-sm font-semibold">Quick Links</h4>
          <ul className="space-y-2.5 text-sm text-white/80">
            {NAV.map((n) => (
              <li key={n.to}><Link className="hover:text-glow" to={n.to}>{n.label}</Link></li>
            ))}
          </ul>
        </div>

        <div>
          <h4 className="mb-4 text-sm font-semibold">Technologies</h4>
          <ul className="space-y-2.5 text-sm text-white/80">
            {TECH.map((t) => (
              <li key={t.name} className="flex items-center gap-2.5">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: t.color }} /> {t.name}
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h4 className="mb-4 text-sm font-semibold">Project Type</h4>
          <ul className="space-y-2.5 text-sm">
            <li className="font-medium text-glow">Deep Learning</li>
            <li className="text-white/80">Mini Project</li>
          </ul>
        </div>
      </div>

      <div className="border-t border-white/10">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-6 py-5 text-xs text-white/70 sm:px-8 md:flex-row md:items-center md:justify-between">
          <span>© {new Date().getFullYear()} DR Vision AI. For educational use only.</span>
          <span className="md:max-w-xs md:text-right">
            This application is for educational and research purposes only. It is not a substitute for professional medical advice.
          </span>
        </div>
      </div>
    </footer>
  )
}
