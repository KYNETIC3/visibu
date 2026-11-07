import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
// Self-hosted so the Content Security Policy (font-src 'self') allows it.
import '@fontsource/atkinson-hyperlegible-next/400.css';
import '@fontsource/atkinson-hyperlegible-next/500.css';
import '@fontsource/atkinson-hyperlegible-next/600.css';
import '@fontsource/atkinson-hyperlegible-next/700.css';
import { App } from './App';
import { purgeDemoTurns } from './lib/history';
import './styles.css';

purgeDemoTurns();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
