import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { partialReview } from './fixtures';

beforeEach(() => {
  window.history.replaceState(null, '', '/');
  vi.stubGlobal('scrollTo', vi.fn());
});

function response(data: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => data };
}

const demoClaims = [
  {
    claim_id: 'DEMO-COHERENT',
    title: 'Coherent claim',
    description: 'Controlled source documents',
  },
];

describe('Review API integration', () => {
  it('lets a visitor choose a case before loading it and refreshes only on request', async () => {
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
    await screen.findByRole('heading', { level: 1, name: 'Explore a claim review' });
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('textbox', { name: 'Open claim' })).not.toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Open Coherent claim' }));
    await screen.findByRole('heading', { level: 1, name: 'Coherent claim' });
    expect(screen.getByRole('heading', { level: 1, name: 'Coherent claim' })).toHaveFocus();
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: 0, behavior: 'instant' });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.mock.calls[1][0]).toBe('/v1/claims/DEMO-COHERENT/review');
    for (const call of fetcher.mock.calls) expect(call[1].method).toBeUndefined();
    await userEvent.click(screen.getByText('Open a claim by ID'));
    await userEvent.click(screen.getByRole('button', { name: 'Refresh stored results' }));
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(3));
    expect(fetcher.mock.calls.every(([url]) => !/assess|agent\/review|\/health/.test(url))).toBe(
      true,
    );
  });

  it('shows a useful missing-claim state and allows retry', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce({ ok: true, json: async () => [] })
        .mockResolvedValue(response({ detail: 'Claim not found' }, 404)),
    );
    render(<App />);
    await screen.findByRole('heading', { name: 'Explore a claim review' });
    expect(screen.getByText('Open a stored claim using its claim ID.')).toBeVisible();
    await userEvent.type(screen.getByRole('textbox', { name: 'Open claim' }), 'UNKNOWN');
    await userEvent.click(screen.getByRole('button', { name: 'Open review' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('This claim was not found.');
    expect(screen.getByRole('button', { name: 'Try again' })).toBeEnabled();
  });

  it('explains a backend connection failure before any claim has been selected', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network failure')));
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Backend unavailable' });
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('http://127.0.0.1:8000');
    expect(alert).toHaveTextContent("$env:CLAIMS_REVIEW_DEMO = '1'");
    expect(alert).toHaveTextContent(
      String.raw`.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000`,
    );
    expect(alert).toHaveTextContent('PostgreSQL and Qdrant are not needed for the prepared demo.');
    expect(alert).toHaveTextContent('it does not start FastAPI');
    expect(
      screen.queryByRole('heading', { name: 'Explore a claim review' }),
    ).not.toBeInTheDocument();
    expect(screen.getByText('Start FastAPI to load the prepared cases.')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Open claim' })).toBeEnabled();
  });

  it.each([502, 504])(
    'handles an HTTP %s gateway failure as an unavailable backend',
    async (status) => {
      vi.stubGlobal(
        'fetch',
        vi
          .fn()
          .mockResolvedValue(
            response(status === 502 ? { error: { code: 'BACKEND_UNREACHABLE' } } : null, status),
          ),
      );
      render(<App />);
      await screen.findByRole('heading', { level: 1, name: 'Backend unavailable' });
      expect(screen.getByRole('button', { name: 'Retry connection' })).toBeEnabled();
      expect(screen.queryByRole('heading', { name: 'Claim not found' })).not.toBeInTheDocument();
    },
  );

  it.each([
    [404, 'Claim not found', 'This claim was not found.'],
    [503, 'Review service unavailable', 'FastAPI is responding'],
    [500, 'The request could not be completed', 'HTTP 500'],
  ])(
    'keeps a live backend HTTP %s distinct from a stopped backend',
    async (status, heading, message) => {
      vi.stubGlobal(
        'fetch',
        vi
          .fn()
          .mockResolvedValueOnce(response([]))
          .mockResolvedValue(response({ detail: 'Controlled API error' }, status as number)),
      );
      render(<App />);
      await screen.findByRole('heading', { name: 'Explore a claim review' });
      await userEvent.type(screen.getByRole('textbox', { name: 'Open claim' }), 'UNKNOWN');
      await userEvent.click(screen.getByRole('button', { name: 'Open review' }));
      await screen.findByRole('heading', { level: 1, name: heading as string });
      expect(screen.getByRole('alert')).toHaveTextContent(message as string);
      expect(
        screen.queryByRole('heading', { name: 'Backend unavailable' }),
      ).not.toBeInTheDocument();
      expect(screen.queryByText('Start the prepared demo backend')).not.toBeInTheDocument();
    },
  );

  it('does not report a catalogue HTTP 404 as a missing claim', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({ detail: 'Not Found' }, 404)));
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'The request could not be completed' });
    expect(screen.getByRole('alert')).toHaveTextContent(
      'The demo list could not be loaded (HTTP 404)',
    );
    expect(screen.queryByText('Start FastAPI to load the prepared cases.')).not.toBeInTheDocument();
    expect(
      screen.getByText('The case list could not be loaded. Check the connection details.'),
    ).toBeVisible();
    expect(screen.queryByText('This claim was not found.')).not.toBeInTheDocument();
  });

  it('recovers case selection after the backend starts and waits for a visitor to choose', async () => {
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(response(demoClaims))
      .mockResolvedValueOnce(response(partialReview()));
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Backend unavailable' });
    await userEvent.click(screen.getByRole('button', { name: 'Retry connection' }));
    await screen.findByRole('heading', { level: 1, name: 'Explore a claim review' });
    expect(fetcher).toHaveBeenCalledTimes(2);
    await userEvent.click(screen.getByRole('button', { name: 'Open Coherent claim' }));
    await screen.findByRole('heading', { level: 1, name: 'Coherent claim' });
    expect(screen.getByRole('button', { name: /Coherent claim/ })).toBeEnabled();
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
      '/v1/review/demo-claims',
      '/v1/review/demo-claims',
      '/v1/claims/DEMO-COHERENT/review',
    ]);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('refreshes the catalogue and preserves the requested claim when retrying an API failure', async () => {
    window.history.replaceState(null, '', '/?claim=DEMO-COHERENT');
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(demoClaims))
      .mockResolvedValueOnce(response({ detail: 'Database unavailable' }, 503))
      .mockResolvedValueOnce(response(demoClaims))
      .mockResolvedValueOnce(response(partialReview()));
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Review service unavailable' });
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
    await screen.findByRole('heading', { level: 1, name: 'Coherent claim' });
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
      '/v1/review/demo-claims',
      '/v1/claims/DEMO-COHERENT/review',
      '/v1/review/demo-claims',
      '/v1/claims/DEMO-COHERENT/review',
    ]);
  });

  it('returns to the case cards without fetching or exposing technical details', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(demoClaims))
      .mockResolvedValue(response(partialReview()));
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { name: 'Explore a claim review' });
    await userEvent.click(screen.getByRole('button', { name: 'Open Coherent claim' }));
    await screen.findByRole('heading', { name: 'Coherent claim' });
    expect(window.location.search).toContain('claim=DEMO-COHERENT');
    await userEvent.click(screen.getByRole('button', { name: 'Back to demo cases' }));
    expect(screen.getByRole('heading', { name: 'Explore a claim review' })).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Explore a claim review' })).toHaveFocus();
    expect(window.location.search).not.toContain('claim=');
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
  });
});
