import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { purgeDemoTurns } from './lib/history'
import './styles.css'

purgeDemoTurns()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
