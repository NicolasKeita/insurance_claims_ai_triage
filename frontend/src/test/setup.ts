import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach } from 'vitest';
import { setLocale } from '../i18n';

beforeEach(() => setLocale('en', false));
afterEach(() => {
  cleanup();
  setLocale('en', false);
});
