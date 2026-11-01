import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { reviewBackendProxy } from './backendProxy';
import { reviewI18n } from './i18nPlugin';
import { reviewStaticDemo } from './staticDemoPlugin';

export default defineConfig(({ mode }) => {
  const staticDemo = mode === 'demo';
  return {
    plugins: [reviewI18n(), react(), ...(staticDemo ? [reviewStaticDemo()] : [])],
    define: staticDemo ? { 'import.meta.env.VITE_REVIEW_RUNTIME': JSON.stringify('static') } : {},
    server: staticDemo ? {} : { proxy: { '/v1': reviewBackendProxy() } },
    preview: staticDemo ? {} : { proxy: { '/v1': reviewBackendProxy() } },
  };
});
