/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';

// Serve the same headers as Vercel (vercel.json) from `npm run preview`, so the
// production policy can be tested locally. The CSP also blocks MediaPipe's
// usage metrics call to odml.pa.googleapis.com.
const vercel = JSON.parse(readFileSync(new URL('./vercel.json', import.meta.url), 'utf8')) as {
  headers: { source: string; headers: { key: string; value: string }[] }[];
};
const productionHeaders = Object.fromEntries(
  vercel.headers.flatMap((rule) => rule.headers).map(({ key, value }) => [key, value]),
);

export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },
  preview: { headers: productionHeaders },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
