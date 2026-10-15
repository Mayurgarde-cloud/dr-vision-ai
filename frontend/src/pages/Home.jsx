import { Link } from 'react-router-dom'
import {
  ArrowRight, BarChart3, BrainCircuit, ChevronDown, HeartPulse, ImagePlus, Layers, ScanSearch,
  ShieldCheck, SlidersHorizontal, Network, GraduationCap, ScanEye, ShieldAlert,
} from 'lucide-react'
import TopNav from '../components/TopNav'
import Footer from '../components/Footer'
import { CLASSES } from '../lib/classes'
import { useTheme } from '../hooks/useTheme'

/* ------------------------------------------------------------------ hero */
function HeroVisual() {
  return (
    <div className="relative mx-auto aspect-square w-[min(82vw,440px)]">
      <div className="absolute -inset-10 rounded-full bg-[radial-gradient(circle,rgba(30,120,255,.35),transparent_65%)]" />
      <div className="absolute inset-0 rounded-full border-2 border-cyan-400/25" />
      <div className="absolute inset-[5%] rounded-full border border-cyan-300/20" />
      <div className="absolute inset-[9%] rounded-full border border-dashed border-cyan-300/15" />
      {/* HUD corner brackets */}
      {[
        'left-[6%] top-[6%] border-l-2 border-t-2',
        'right-[6%] top-[6%] border-r-2 border-t-2',
        'left-[6%] bottom-[6%] border-b-2 border-l-2',
        'right-[6%] bottom-[6%] border-b-2 border-r-2',
      ].map((c) => (
        <span key={c} className={`absolute h-10 w-10 border-cyan-300/50 ${c}`} />
      ))}
      <img
        src="/samples/hero.png"
        alt="Illustrative retinal fundus image"
        className="absolute inset-[8%] h-[84%] w-[84%] rounded-full object-cover shadow-[0_0_60px_rgba(255,120,30,.35)]"
      />
      <div className="absolute inset-[8%] rounded-full ring-1 ring-white/10" />
    </div>
  )
}

function Hero() {
  const chips = [
    { icon: BrainCircuit, title: 'Deep Learning', sub: 'ResNet-50 + DenseNet-121' },
    { icon: ScanSearch, title: 'Image Analysis', sub: 'Fundus Image Processing' },
    { icon: Layers, title: '5-Class Classification', sub: '0 - 4 Severity Levels' },
  ]
  return (
    <section className="relative overflow-hidden bg-hero text-white">
      <TopNav />
      <div className="mx-auto grid max-w-7xl items-center gap-10 px-6 pb-14 pt-8 sm:px-8 lg:grid-cols-[1.2fr_1fr] lg:pb-16 lg:pt-12">
        <div className="fade-up">
          <h1 className="text-[34px] font-extrabold leading-[1.1] tracking-tight sm:text-[44px] lg:text-[48px]">
            Diabetic Retinopathy
            <span className="block text-glow sm:whitespace-nowrap">Severity Classification</span>
          </h1>
          <p className="mt-5 max-w-md text-[15px] leading-relaxed text-sky-200/90">
            AI-powered educational screening application that analyses retinal fundus images using deep learning.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <Link to="/predict" className="btn bg-glow px-8 py-3.5 text-navy-950 shadow-glow hover:bg-cyan-300">
              Start Prediction
            </Link>
            <a href="#about-project" className="btn border border-white/40 px-8 py-3.5 text-white hover:bg-white/10">
              Explore Project
            </a>
          </div>
          <div className="mt-10 grid gap-5 sm:grid-cols-3">
            {chips.map(({ icon: Icon, title, sub }) => (
              <div key={title} className="flex items-center gap-3">
                <span className="grid h-12 w-12 shrink-0 place-items-center rounded-full border border-white/10 bg-white/10 text-cyan-200">
                  <Icon size={21} />
                </span>
                <div className="leading-tight">
                  <div className="text-[13.5px] font-semibold text-glow">{title}</div>
                  <div className="mt-0.5 text-[11.5px] text-white/65">{sub}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <HeroVisual />
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ sections */
function SectionTitle({ children }) {
  return (
    <div className="mb-5 flex items-center gap-4">
      <h2 className="text-xl font-bold text-ink">{children}</h2>
      <span className="h-px flex-1 bg-gradient-to-r from-line to-transparent" />
    </div>
  )
}

function WhatIsDR() {
  const cards = [
    { icon: HeartPulse, title: 'Early Detection', sub: 'Better Outcomes' },
    { icon: Network, title: 'AI Analysis', sub: 'Fast & Automated' },
    { icon: GraduationCap, title: 'Educational Use', sub: 'Not a Clinical Diagnosis' },
  ]
  return (
    <section id="about-project" className="grid items-center gap-6 lg:grid-cols-[1.25fr_1fr]">
      <div className="flex items-center gap-5">
        <img src="/samples/thumb.jpg" alt="Retinal fundus" className="h-24 w-24 shrink-0 rounded-2xl bg-black object-cover shadow-card sm:h-28 sm:w-28" />
        <div>
          <h2 className="text-xl font-bold text-ink">What is Diabetic Retinopathy?</h2>
          <p className="mt-2 max-w-md text-sm leading-relaxed text-muted">
            Diabetic retinopathy is a leading cause of vision loss. Early detection and severity assessment can help prevent progression and preserve vision.
          </p>
          <Link to="/about" className="mt-3 inline-flex items-center gap-1 text-sm font-semibold text-brand-600 hover:underline dark:text-brand-300">
            Learn More <ArrowRight size={14} />
          </Link>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3">
        {cards.map(({ icon: Icon, title, sub }) => (
          <div key={title} className="card flex flex-col items-center gap-2 px-2 py-4 text-center">
            <Icon size={24} className="text-brand-500" />
            <div className="text-[13px] font-bold text-ink">{title}</div>
            <div className="text-[11px] text-muted">{sub}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function HowItWorks() {
  const steps = [
    { icon: ImagePlus, title: 'Upload Image', sub: 'JPG, JPEG, PNG' },
    { icon: SlidersHorizontal, title: 'Preprocess', sub: 'Resize & Normalize' },
    { icon: ScanSearch, title: 'Extract Features', sub: 'ResNet-50 + DenseNet-121' },
    { icon: ShieldCheck, title: 'Classify', sub: 'PCA + Ensemble' },
    { icon: BarChart3, title: 'Show Result', sub: 'Confidence & Grad-CAM' },
  ]
  return (
    <section>
      <SectionTitle>How It Works</SectionTitle>
      <ol className="card grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-5 lg:gap-0">
        {steps.map(({ icon: Icon, title, sub }, i) => (
          <li key={title} className="relative flex items-center gap-3 lg:px-3">
            <span className="grid h-14 w-14 shrink-0 place-items-center rounded-full bg-soft text-brand-500 ring-4 ring-soft/60">
              <Icon size={24} />
            </span>
            <div className="leading-tight">
              <div className="text-[13.5px] font-bold text-ink">{title}</div>
              <div className="mt-1 text-[11px] text-muted">{sub}</div>
            </div>
            {i < steps.length - 1 && <ArrowRight size={16} className="absolute -right-2 hidden text-brand-400 lg:block" />}
          </li>
        ))}
      </ol>
    </section>
  )
}

function SeverityClasses() {
  const { dark } = useTheme()
  return (
    <section>
      <SectionTitle>Five Severity Classes</SectionTitle>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {CLASSES.map((c) => (
          <article
            key={c.index}
            className="rounded-2xl border p-3.5 shadow-card"
            style={{
              borderColor: `${c.color}55`,
              background: `linear-gradient(180deg, ${dark ? c.darkSoft : c.soft}, var(--card) 60%)`,
            }}
          >
            <div className="mb-3 flex items-center gap-2.5">
              <span className="grid h-7 w-7 place-items-center rounded-lg text-sm font-bold text-white" style={{ background: c.color }}>
                {c.index}
              </span>
              <div className="leading-tight">
                <div className="text-sm font-bold text-ink">{c.name}</div>
                <div className="text-[11px] text-muted">{c.short}</div>
              </div>
            </div>
            <img
              src={`/samples/class_${c.index}.jpg`}
              alt={`Illustrative ${c.name} fundus`}
              loading="lazy"
              className="aspect-square w-full rounded-xl bg-black object-cover"
            />
          </article>
        ))}
      </div>
      <p className="mt-3 text-[11.5px] text-muted">Images are synthetic illustrations generated for this project, not patient photographs.</p>
    </section>
  )
}

function Arrow() {
  return (
    <>
      <ArrowRight size={18} className="hidden shrink-0 text-brand-400 lg:block" />
      <ChevronDown size={18} className="shrink-0 text-brand-400 lg:hidden" />
    </>
  )
}

function Box({ children, className = '' }) {
  return (
    <div className={`grid place-items-center rounded-xl border border-line bg-soft px-5 py-3 text-center text-[13px] font-bold text-brand-700 shadow-sm dark:text-brand-300 ${className}`}>
      {children}
    </div>
  )
}

function Architecture() {
  return (
    <section>
      <SectionTitle>Model Architecture</SectionTitle>
      <div className="card flex flex-col items-center gap-3 p-5 lg:flex-row lg:justify-between lg:gap-2">
        <div className="flex w-36 flex-col items-center gap-2 rounded-2xl border border-line bg-card2 p-3">
          <img src="/samples/thumb.jpg" alt="" className="h-16 w-16 rounded-full bg-black object-cover" />
          <span className="text-[11.5px] text-muted">Retinal Image</span>
        </div>
        <Arrow />
        <div className="flex w-44 flex-col gap-3">
          <Box>ResNet-50</Box>
          <Box>DenseNet-121</Box>
        </div>
        <Arrow />
        <Box className="w-36 py-7">Feature<br />Fusion</Box>
        <Arrow />
        <Box className="w-28 py-6">PCA</Box>
        <Arrow />
        <Box className="w-40 py-5">AdaBoost<br />+<br />Random Forest</Box>
        <Arrow />
        <Box className="w-36 py-5 !bg-card2 !text-ink">DR Severity<br />(0 - 4)</Box>
      </div>
    </section>
  )
}

function CTA() {
  return (
    <section className="flex flex-col gap-5 rounded-2xl bg-gradient-to-r from-navy-900 to-navy-700 p-6 text-white shadow-card md:flex-row md:items-center md:justify-between">
      <div className="flex items-center gap-4">
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-glow/15 text-glow ring-1 ring-glow/40">
          <ScanEye size={22} />
        </span>
        <div>
          <h3 className="text-lg font-bold">Ready to Analyse Your Retinal Image?</h3>
          <p className="mt-1 text-sm text-sky-200/80">Upload a retinal fundus image and get instant AI-powered prediction with confidence scores.</p>
        </div>
      </div>
      <Link to="/predict" className="btn shrink-0 bg-gradient-to-b from-brand-400 to-brand-600 px-8 text-white shadow-btn">
        Start Prediction <ArrowRight size={16} />
      </Link>
      <div className="flex items-start gap-3 border-white/10 text-xs leading-relaxed text-sky-200/80 md:max-w-[230px] md:border-l md:pl-5">
        <ShieldAlert size={16} className="mt-0.5 shrink-0 text-glow" />
        This application is for educational and research purposes only. It is not a substitute for professional medical advice.
      </div>
    </section>
  )
}

export default function Home() {
  return (
    <div className="bg-navy-900">
      <Hero />
      <div className="bg-page">
        <div className="mx-auto max-w-6xl space-y-10 px-6 py-10 sm:px-8">
          <WhatIsDR />
          <HowItWorks />
          <SeverityClasses />
          <Architecture />
          <CTA />
        </div>
      </div>
      <Footer />
    </div>
  )
}
