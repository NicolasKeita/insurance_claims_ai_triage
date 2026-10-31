import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import { reviewI18n } from './i18nPlugin';

export default defineConfig({
  plugins: [reviewI18n(), react()],
  test: { environment: 'jsdom', setupFiles: ['./src/test/setup.ts'], clearMocks: true },
});
