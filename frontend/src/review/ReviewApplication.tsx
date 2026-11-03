'use client';

import { useEffect } from 'react';
import App from '../App';
import { I18nProvider, initializeI18n } from '../i18n';
import { initializeTheme } from '../theme';

export function ReviewApplication() {
  useEffect(() => {
    initializeTheme();
    initializeI18n();
  }, []);

  return (
    <I18nProvider>
      <App />
    </I18nProvider>
  );
}
