import { useState } from 'react'
import { ChevronDown, ExternalLink } from 'lucide-react'
import { PageHeader } from '../components/ui'

const SECTIONS = [
  {
    title: 'Problem Statement',
    body: (
      <p>
        Diabetic retinopathy (DR) damages the blood vessels of the retina and is a leading cause of preventable vision loss.
        Grading severity from fundus photographs by hand is slow and needs trained specialists. This project explores whether
        deep features from pre-trained CNNs, combined with a boosted ensemble, can grade DR severity (0–4) automatically.
      </p>
    ),
  },
  {
    title: 'Objectives',
    body: (
      <ul className="list-disc space-y-1 pl-5">
        <li>Classify retinal fundus images into five DR severity grades.</li>
        <li>Compare and fuse complementary features from ResNet-50 and DenseNet-121.</li>
        <li>Reduce dimensionality with PCA and classify with AdaBoost over Random Forests.</li>
        <li>Explain predictions with class-activation heat-maps and report standard metrics.</li>
      </ul>
    ),
  },
  {
    title: 'Methodology',
    body: (
      <ol className="list-decimal space-y-1 pl-5">
        <li>Crop the fundus, apply CLAHE contrast enhancement, resize to 224 × 224 and normalise with ImageNet statistics.</li>
        <li>Extract 2048-d (ResNet-50) and 1024-d (DenseNet-121) global-pooled features and concatenate them (3072-d).</li>
        <li>Standardise, reduce with PCA, then classify with an AdaBoost ensemble whose weak learners are Random Forests.</li>
        <li>Evaluate on a stratified 70 / 15 / 15 train / validation / test split (accuracy, macro precision / recall / F1, confusion matrix).</li>
        <li>Visualise the evidence with a class-activation map computed from the ResNet-50 layer-4 features.</li>
      </ol>
    ),
  },
  {
    title: 'Future Improvements',
    body: (
      <ul className="list-disc space-y-1 pl-5">
        <li>Fine-tune the backbones end-to-end instead of using frozen features.</li>
        <li>Train on larger public datasets (EyePACS, APTOS 2019, Messidor-2) and validate across datasets.</li>
        <li>Add lesion segmentation, uncertainty estimation and probability calibration.</li>
        <li>Clinical validation with ophthalmologists before any real-world use.</li>
      </ul>
    ),
  },
]

const TECH = [
  ['Python', '#f5c518'], ['PyTorch', '#ee4c2c'], ['FastAPI', '#05c8a7'], ['React + Vite', '#61dafb'],
  ['OpenCV', '#2f9e44'], ['scikit-learn', '#3b9ddd'], ['Pandas / NumPy', '#7c5cff'], ['Recharts', '#e8590c'],
  ['SQLite', '#2b7fd6'], ['Tailwind CSS', '#22b8cf'],
]

function Accordion({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card overflow-hidden">
      <button
        className="flex w-full items-center gap-3 px-5 py-3.5 text-left text-sm font-semibold text-ink"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="h-2 w-2 rounded-full bg-brand-500" />
        <span className="flex-1">{title}</span>
        <ChevronDown size={18} className={`text-brand-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && <div className="border-t border-line px-5 py-4 text-[13.5px] leading-relaxed text-muted">{children}</div>}
    </div>
  )
}

export default function About() {
  return (
    <div className="fade-up">
      <PageHeader title="About DR Vision AI" subtitle="Diabetic Retinopathy Severity Classification Using Deep Learning" />

      <div className="grid gap-6 lg:grid-cols-[1.7fr_1fr]">
        <div className="space-y-4">
          <section className="card flex flex-col gap-5 p-5 sm:flex-row">
            <img src="/samples/class_2.jpg" alt="Illustrative fundus" className="h-36 w-36 shrink-0 rounded-xl bg-black object-cover" />
            <p className="text-[13.5px] leading-relaxed text-muted">
              DR Vision AI is a deep-learning-based educational application that analyses retinal fundus images to classify the
              severity of diabetic retinopathy. This project is developed as a mini project for the Deep Learning subject and is
              not a clinically validated diagnostic tool.
            </p>
          </section>
          {SECTIONS.map((s, i) => (
            <Accordion key={s.title} title={s.title} defaultOpen={i === 0}>{s.body}</Accordion>
          ))}
        </div>

        <div className="space-y-5">
          <section className="card p-5">
            <h2 className="panel-title mb-4 text-base">Technology Stack</h2>
            <ul className="grid grid-cols-2 gap-x-4 gap-y-3.5 text-[13px] text-muted">
              {TECH.map(([name, color]) => (
                <li key={name} className="flex items-center gap-2.5">
                  <span className="h-3 w-3 shrink-0 rounded-[4px]" style={{ background: color }} /> {name}
                </li>
              ))}
            </ul>
          </section>

          <section className="card p-5">
            <h2 className="panel-title mb-3 text-[13px]">Research Paper Inspiration</h2>
            <a
              href="https://doi.org/10.1109/ACCESS.2022.3217216"
              target="_blank"
              rel="noreferrer noopener"
              className="group text-sm font-semibold leading-snug text-brand-600 hover:underline dark:text-brand-300"
            >
              Multi-Stream Deep Neural Network for Diabetic Retinopathy Severity Classification Under a Boosting Framework
              <ExternalLink size={13} className="ml-1 inline align-[-1px] opacity-70" />
            </a>
            <p className="mt-2 text-xs text-muted">Mustafa et al. · IEEE Access, 2022</p>
          </section>
        </div>
      </div>
    </div>
  )
}
