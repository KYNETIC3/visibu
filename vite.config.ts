/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';

// The same Content Security Policy is served by Vercel (vercel.json) and by
// `npm run preview`, so the production policy can be tested locally. It also
// blocks MediaPipe's usage metrics call to odml.pa.googleapis.com.
const csp = JSON.parse(readFileSync(new URL('./vercel.json', import.meta.url), 'utf8')).headers[0].headers.find(
  (h: { key: string }) => h.key === 'Content-Security-Policy',
).value as string;

export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },
  preview: { headers: { 'Content-Security-Policy': csp } },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
