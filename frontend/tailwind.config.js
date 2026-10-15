/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['"Inter Variable"', 'Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'] },
      colors: {
        navy: { 950: '#040f26', 900: '#061a3f', 800: '#0a2455', 700: '#0e316f', 600: '#154396' },
        brand: { DEFAULT: '#0b7ff0', 700: '#0a62c4', 600: '#0b72dc', 500: '#1389ff', 400: '#27a8ff', 300: '#62c6ff' },
        glow: '#19c3ff',
        ink: 'var(--ink)',
        muted: 'var(--muted)',
        page: 'var(--page)',
        card: 'var(--card)',
        card2: 'var(--card2)',
        line: 'var(--line)',
        soft: 'var(--soft)',
      },
      boxShadow: {
        card: '0 10px 30px -16px rgba(10, 70, 150, 0.35), 0 1px 0 rgba(255,255,255,0.7) inset',
        glow: '0 8px 24px -6px rgba(25, 195, 255, 0.55)',
        btn: '0 8px 20px -8px rgba(11, 127, 240, 0.7)',
      },
      backgroundImage: {
        hero: 'radial-gradient(1200px 520px at 78% 40%, rgba(21,110,230,.45), transparent 60%), radial-gradient(700px 380px at 8% 0%, rgba(25,195,255,.18), transparent 60%), linear-gradient(180deg,#041434 0%,#07225a 55%,#0a2f78 100%)',
      },
    },
  },
  plugins: [],
}
