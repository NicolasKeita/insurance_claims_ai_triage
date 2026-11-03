import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ErrorPanel } from '../components/ErrorPanel';
import { setLocale } from '../i18n';
import type { Locale } from '../i18n';
import { RequestFailure } from '../request';

beforeEach(() => vi.stubEnv('NEXT_PUBLIC_REVIEW_RUNTIME', 'docker'));
afterEach(() => vi.unstubAllEnvs());

describe('Docker connection guidance', () => {
  it.each([
    { locale: 'en', heading: 'Start the Docker demo', retry: 'Retry connection' },
    { locale: 'fr', heading: 'Démarrer la démo Docker', retry: 'Réessayer la connexion' },
  ])(
    'explains API container startup in $locale and retains retry',
    async ({ locale, heading, retry }) => {
      setLocale(locale as Locale, false);
      const onRetry = vi.fn();
      render(
        <ErrorPanel
          error={new RequestFailure('backend-unavailable', undefined, 'catalog')}
          onRetry={onRetry}
        />,
      );
      const alert = screen.getByRole('alert');
      expect(screen.getByRole('heading', { level: 2, name: heading })).toBeVisible();
      expect(alert).toHaveTextContent('docker compose up --build -d frontend_demo');
      expect(alert).toHaveTextContent('docker compose logs api');
      expect(alert).toHaveTextContent('analysis_results');
      expect(alert).not.toHaveTextContent('http://127.0.0.1:8000');
      expect(alert).not.toHaveTextContent('.venv');
      expect(alert).not.toHaveTextContent('it does not start FastAPI');
      await userEvent.click(screen.getByRole('button', { name: retry }));
      expect(onRetry).toHaveBeenCalledOnce();
    },
  );

  it.each(['en', 'fr'] as const)(
    'directs a service failure to the mounted archive and API logs in %s',
    (locale) => {
      setLocale(locale, false);
      render(
        <ErrorPanel
          error={new RequestFailure('service-unavailable', 503, 'claim')}
          onRetry={vi.fn()}
        />,
      );
      const alert = screen.getByRole('alert');
      expect(alert).toHaveTextContent('analysis_results/manifest.json');
      expect(alert).toHaveTextContent('docker compose logs api');
      expect(alert).not.toHaveTextContent('DATABASE_URL');
      expect(alert).not.toHaveTextContent('PostgreSQL');
      expect(alert).not.toHaveTextContent('.venv');
    },
  );
});
