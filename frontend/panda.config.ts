import { defineConfig } from '@pandacss/dev';
import { reviewStyles } from './panda/global-styles';
import { reviewTheme, reviewTokens } from './panda/theme';

export default defineConfig({
  preflight: false,
  presets: ['@pandacss/preset-base', '@pandacss/preset-panda'],
  include: ['./src/**/*.{ts,tsx}'],
  exclude: ['./src/test/**'],
  outdir: 'styled-system',
  prefix: 'claims',
  globalCss: { ...reviewTheme, ...reviewStyles },
  theme: {
    extend: {
      tokens: reviewTokens,
    },
  },
});
