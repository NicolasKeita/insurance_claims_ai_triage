import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { partialReview } from './fixtures';

beforeEach(() => window.history.replaceState(null, '', '/'));

describe('Review API integration', () => {
  it('loads a demo review using read-only requests and refreshes stored data explicitly', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          {
            claim_id: 'DEMO-COHERENT',
            title: 'Coherent claim',
            description: 'Controlled source documents',
          },
        ],
      })
      .mockResolvedValue({ ok: true, json: async () => partialReview() });
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading claim review');
    await screen.findByRole('heading', { level: 1, name: 'DEMO-COHERENT' });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.mock.calls[1][0]).toBe('/v1/claims/DEMO-COHERENT/review');
    for (const call of fetcher.mock.calls) expect(call[1].method).toBeUndefined();
    await userEvent.click(screen.getByRole('button', { name: 'Refresh stored results' }));
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(3));
    expect(fetcher.mock.calls.every(([url]) => !/assess|agent\/review/.test(url))).toBe(true);
  });

  it('shows a useful missing-claim state and allows retry', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce({ ok: true, json: async () => [] })
        .mockResolvedValue({ ok: false, status: 404 }),
    );
    render(<App />);
    await screen.findByRole('heading', { name: /Every claim has a story/ });
    await userEvent.type(screen.getByRole('textbox', { name: 'Open claim' }), 'UNKNOWN');
    await userEvent.click(screen.getByRole('button', { name: 'Open review' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('This claim was not found.');
    expect(screen.getByRole('button', { name: 'Try again' })).toBeEnabled();
  });

  it('handles an unavailable demo list without hiding manual claim access', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network failure')));
    render(<App />);
    await screen.findByRole('heading', { name: /Every claim has a story/ });
    expect(
      screen.getByText('Demo list unavailable. You can open a claim by ID.'),
    ).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Open claim' })).toBeEnabled();
  });
});
