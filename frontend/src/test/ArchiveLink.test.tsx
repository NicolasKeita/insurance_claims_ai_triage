import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { partialReview } from './fixtures';

const claims = [
  { claim_id: 'E2E-VISUAL-0001', title: 'Visual claim', description: 'Recorded photo evidence' },
  { claim_id: 'E2E-QUOTE-0001', title: 'Quote claim', description: 'Recorded document evidence' },
];

function response(data: unknown) {
  return { ok: true, json: async () => data };
}

function savedReview(claimId = 'E2E-VISUAL-0001') {
  const review = partialReview();
  review.claim.claim_id = claimId;
  return review;
}

beforeEach(() => {
  window.history.replaceState(null, '', '/?claim=E2E-VISUAL-0001&analysis=ANALYSIS-0003');
  vi.stubGlobal('scrollTo', vi.fn());
});

describe('Immutable analysis links', () => {
  it('loads and refreshes the selected analysis and keeps it through language changes', async () => {
    const review = savedReview();
    const original = JSON.stringify(review);
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(claims))
      .mockResolvedValue(response(review));
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Front collision' });
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
      '/v1/review/demo-claims',
      '/v1/claims/E2E-VISUAL-0001/review?analysis=ANALYSIS-0003',
    ]);
    expect(new URL(window.location.href).searchParams.get('analysis')).toBe('ANALYSIS-0003');

    await userEvent.click(screen.getByRole('button', { name: 'Français' }));
    await screen.findByRole('heading', { level: 1, name: 'Collision à l’avant du véhicule' });
    await userEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(new URL(window.location.href).searchParams.get('analysis')).toBe('ANALYSIS-0003');

    await userEvent.click(screen.getByText('Open a claim by ID'));
    await userEvent.click(screen.getByRole('button', { name: 'Refresh stored results' }));
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(3));
    expect(fetcher.mock.calls[2][0]).toBe(
      '/v1/claims/E2E-VISUAL-0001/review?analysis=ANALYSIS-0003',
    );
    expect(new URL(window.location.href).searchParams.get('analysis')).toBe('ANALYSIS-0003');
    expect(JSON.stringify(review)).toBe(original);
  });

  it('clears the analysis when selecting another case', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(claims))
      .mockResolvedValueOnce(response(savedReview()))
      .mockResolvedValueOnce(response(savedReview('E2E-QUOTE-0001')));
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Front collision' });
    await userEvent.click(screen.getByText('Try another case'));
    await userEvent.click(screen.getByRole('button', { name: 'Open Extra repair in the quote' }));
    await screen.findByRole('heading', { level: 1, name: 'Extra repair in the quote' });
    expect(fetcher.mock.calls[2][0]).toBe('/v1/claims/E2E-QUOTE-0001/review');
    const selection = new URL(window.location.href).searchParams;
    expect(selection.get('claim')).toBe('E2E-QUOTE-0001');
    expect(selection.has('analysis')).toBe(false);
  });

  it('removes the claim and analysis selection when returning to the case catalogue', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(claims))
      .mockResolvedValueOnce(response(savedReview()));
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Front collision' });
    await userEvent.click(screen.getByRole('button', { name: 'Explore other cases' }));
    expect(screen.getByRole('heading', { name: 'Explore a claim review' })).toBeVisible();
    const selection = new URL(window.location.href).searchParams;
    expect(selection.has('claim')).toBe(false);
    expect(selection.has('analysis')).toBe(false);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('does not apply an analysis selection without a matching explicit claim', async () => {
    window.history.replaceState(null, '', '/?analysis=ANALYSIS-0003');
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(claims))
      .mockResolvedValueOnce(response(savedReview()));
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Front collision' });
    expect(fetcher.mock.calls[1][0]).toBe('/v1/claims/E2E-VISUAL-0001/review');
    expect(new URL(window.location.href).searchParams.has('analysis')).toBe(false);
  });
});
