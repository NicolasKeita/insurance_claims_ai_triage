import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { reviewBackendProxy } from './backendProxy';
import { reviewI18n } from './i18nPlugin';

export default defineConfig({
  plugins: [reviewI18n(), react()],
  server: { proxy: { '/v1': reviewBackendProxy() } },
  preview: { proxy: { '/v1': reviewBackendProxy() } },
});
