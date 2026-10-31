import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { LOCALE_STORAGE_KEY, initializeI18n } from '../i18n';
import { partialReview } from './fixtures';

beforeEach(() => {
  const values = new Map<string, string>();
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
  });
  vi.stubGlobal('scrollTo', vi.fn());
  window.history.replaceState(null, '', '/');
  initializeI18n();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('Bilingual prepared review', () => {
  it('starts in French and changes presentation and workflow image without refetching the case', async () => {
    const review = partialReview();
    review.claim.claim_id = 'E2E-VISUAL-0001';
    const original = JSON.stringify(review);
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          {
            claim_id: 'E2E-VISUAL-0001',
            title: 'Visual additional damage',
            description: 'Recorded evidence',
          },
        ],
      })
      .mockResolvedValueOnce({ ok: true, json: async () => review });
    vi.stubGlobal('fetch', fetcher);
    const { container } = render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Analyse photo' });
    expect(document.documentElement).toHaveAttribute('lang', 'fr');
    expect(screen.getByRole('button', { name: 'Français' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    const workflow = () =>
      Array.from(container.querySelectorAll('img')).find((image) =>
        image.src.includes('review-workflow-'),
      );
    expect(workflow()).toHaveAttribute('src', expect.stringContaining('review-workflow-fr-v2.png'));
    expect(workflow()?.alt).not.toBe('');
    expect(container.textContent?.replace(/[\u00a0\u202f]/g, ' ')).toContain('1 240,00 €');
    expect(container).toHaveTextContent('20 janv. 2026');
    expect(fetcher).toHaveBeenCalledTimes(2);

    await userEvent.click(screen.getByRole('button', { name: 'English' }));
    await screen.findByRole('heading', { level: 1, name: 'Photo evidence' });
    expect(document.documentElement).toHaveAttribute('lang', 'en');
    expect(screen.getByRole('button', { name: 'English' })).toHaveAttribute('aria-pressed', 'true');
    expect(window.localStorage.getItem(LOCALE_STORAGE_KEY)).toBe('en');
    expect(workflow()).toHaveAttribute('src', expect.stringContaining('review-workflow-en-v2.png'));
    expect(container).toHaveTextContent('€1,240.00');
    expect(container).toHaveTextContent('20 Jan 2026');
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));

    await userEvent.click(screen.getByRole('button', { name: 'Français' }));
    await screen.findByRole('heading', { level: 1, name: 'Analyse photo' });
    expect(document.documentElement).toHaveAttribute('lang', 'fr');
    expect(workflow()).toHaveAttribute('src', expect.stringContaining('review-workflow-fr-v2.png'));
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
      '/v1/review/demo-claims',
      '/v1/claims/E2E-VISUAL-0001/review',
    ]);
    expect(JSON.stringify(review)).toBe(original);
  });
});
