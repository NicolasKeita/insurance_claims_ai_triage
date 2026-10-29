import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { reviewBackendProxy } from './backendProxy';

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/v1': reviewBackendProxy() } },
  preview: { proxy: { '/v1': reviewBackendProxy() } },
});
