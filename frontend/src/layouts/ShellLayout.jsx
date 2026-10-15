
import { Outlet } from 'react-router-dom'
import TopNav from '../components/TopNav'
import Sidebar from '../components/Sidebar'
import DemoBanner from '../components/DemoBanner'

export default function ShellLayout() {
  return (
    <div className="flex min-h-screen flex-col bg-navy-900">
      <TopNav />

      <div className="flex flex-1">
        <Sidebar />

        <main className="flex min-w-0 flex-1 flex-col bg-page lg:rounded-tl-[28px]">
          <div className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-8 sm:py-9">
            <DemoBanner />
            <Outlet />
          </div>

          <p className="px-6 pb-5 text-center text-[11.5px] text-muted">
            © {new Date().getFullYear()} DR Vision AI · For educational and
            research purposes only. Not a substitute for professional medical
            advice.
          </p>
        </main>
      </div>
    </div>
  )
}