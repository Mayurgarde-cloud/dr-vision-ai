import { useCallback, useEffect, useRef, useState } from 'react'

/** Run an async loader on mount / when deps change. Returns { data, error, loading, reload }. */
export function useFetch(loader, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const alive = useRef(true)
  const loaderRef = useRef(loader)
  loaderRef.current = loader

  const run = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }))
    try {
      const data = await loaderRef.current()
      if (alive.current) setState({ data, error: null, loading: false })
    } catch (e) {
      if (alive.current) setState({ data: null, error: e.message || String(e), loading: false })
    }
  }, [])

  useEffect(() => {
    alive.current = true
    run()
    return () => { alive.current = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { ...state, reload: run }
}
