import { createBrowserRouter, Navigate, Outlet, ScrollRestoration } from 'react-router-dom'
import ShellLayout from './layouts/ShellLayout'
import Home from './pages/Home'
import Predict from './pages/Predict'
import Results from './pages/Results'
import Dashboard from './pages/Dashboard'
import Dataset from './pages/Dataset'
import ModelPerformance from './pages/ModelPerformance'
import History from './pages/History'
import About from './pages/About'

function Root() {
  return (
    <>
      <Outlet />
      <ScrollRestoration />
    </>
  )
}

export const router = createBrowserRouter([
  {
    element: <Root />,
    children: [
      { path: '/', element: <Home /> },
      {
        element: <ShellLayout />,
        children: [
          { path: '/predict', element: <Predict /> },
          { path: '/results/:id', element: <Results /> },
          { path: '/dashboard', element: <Dashboard /> },
          { path: '/dataset', element: <Dataset /> },
          { path: '/performance', element: <ModelPerformance /> },
          { path: '/history', element: <History /> },
          { path: '/about', element: <About /> },
        ],
      },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])
