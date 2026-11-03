import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { setLocale } from '../i18n';
import { available, partialReview } from './fixtures';

const claims = [
  { claim_id: 'E2E-VISUAL-0001', title: 'Visual claim', description: 'Recorded photos' },
  { claim_id: 'E2E-QUOTE-0001', title: 'Quote claim', description: 'Recorded documents' },
];
const manifest = {
  format: 'claims_static_demo_v1',
  claims: claims.map((claim) => ({
    claim_id: claim.claim_id,
    analysis_id: 'ANALYSIS-01',
    review: `reviews/${claim.claim_id}_ANALYSIS-01.json`,
  })),
};

function reviewFor(id: string) {
  const review = partialReview();
  review.claim.claim_id = id;
  review.claim.images = ['source.jpg'];
  review.documents.data!.items[0].document_url = 'demo/documents/CASE_ANALYSIS/claim_form.pdf';
  review.vision = available({
    images: [
      {
        filename: 'source.jpg',
        image_url: 'demo/images/CASE_ANALYSIS/source.jpg',
        evidence: [],
      },
    ],
  });
  return review;
}

function response(value: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => value };
}

function fileServer() {
  return vi.fn(async (url: string) => {
    if (url === '/portfolio/demo/catalog.json') return response(claims);
    if (url === '/portfolio/demo/manifest.json') return response(manifest);
    const selected = manifest.claims.find((entry) => url === `/portfolio/demo/${entry.review}`);
    return selected ? response(reviewFor(selected.claim_id)) : response({}, 404);
  });
}

beforeEach(() => {
  vi.stubEnv('NEXT_PUBLIC_REVIEW_RUNTIME', 'static');
  vi.stubEnv('NEXT_PUBLIC_REVIEW_BASE_PATH', '/portfolio/');
  vi.stubGlobal('scrollTo', vi.fn());
  window.history.replaceState(null, '', '/portfolio/');
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe('Standalone static demo', () => {
  it('opens a saved review, its copied PDF and photographs under a hosting subpath', async () => {
    const fetcher = fileServer();
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Front collision' });
    expect(screen.getByRole('link', { name: 'Insurance AI home' })).toHaveAttribute(
      'href',
      '/portfolio/',
    );
    expect(screen.getByRole('img', { name: 'Source claim photograph preview' })).toHaveAttribute(
      'src',
      '/portfolio/demo/images/CASE_ANALYSIS/source.jpg',
    );
    await userEvent.click(screen.getByRole('tab', { name: 'Evidence' }));
    await userEvent.click(screen.getByRole('button', { name: 'Documents' }));
    expect(screen.getByRole('link', { name: 'Open PDF: claim_form.pdf' })).toHaveAttribute(
      'href',
      '/portfolio/demo/documents/CASE_ANALYSIS/claim_form.pdf',
    );
    await userEvent.click(screen.getByRole('button', { name: 'Photos' }));
    expect(screen.getByRole('img', { name: 'Claim photograph: source.jpg' })).toHaveAttribute(
      'src',
      '/portfolio/demo/images/CASE_ANALYSIS/source.jpg',
    );
    expect(fetcher.mock.calls.every(([url]) => url.startsWith('/portfolio/demo/'))).toBe(true);
  });

  it('switches cases without a backend and clears the previous analysis selection', async () => {
    window.history.replaceState(null, '', '/portfolio/?claim=E2E-VISUAL-0001&analysis=ANALYSIS-01');
    const fetcher = fileServer();
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: 'Front collision' });
    await userEvent.click(screen.getByText('Try another case'));
    await userEvent.click(screen.getByRole('button', { name: 'Open Extra repair in the quote' }));
    await screen.findByRole('heading', { level: 1, name: 'Extra repair in the quote' });
    expect(new URLSearchParams(window.location.search).get('claim')).toBe('E2E-QUOTE-0001');
    expect(new URLSearchParams(window.location.search).has('analysis')).toBe(false);
    expect(fetcher.mock.calls.every(([url]) => !url.includes('/v1/'))).toBe(true);
  });

  it.each(['UNKNOWN', 'UNKNOWN-ANALYSIS'])(
    'does not substitute another analysis for an unknown link: %s',
    async (value) => {
      window.history.replaceState(null, '', `/portfolio/?claim=E2E-VISUAL-0001&analysis=${value}`);
      vi.stubGlobal('fetch', fileServer());
      render(<App />);
      await screen.findByRole('heading', { name: 'Claim not found' });
      expect(screen.getByRole('alert')).toHaveTextContent(
        'The selected analysis is not included in this demo.',
      );
      expect(screen.getByRole('alert')).not.toHaveTextContent('FastAPI');
    },
  );

  it.each([
    ['en', 'Demo unavailable', 'Try again'],
    ['fr', 'Démonstration indisponible', 'Réessayer'],
  ] as const)(
    'offers a file retry without API startup advice in %s',
    async (locale, heading, retry) => {
      setLocale(locale, false);
      const server = fileServer();
      const fetcher = vi
        .fn()
        .mockRejectedValueOnce(new TypeError('File unavailable'))
        .mockImplementation(server);
      vi.stubGlobal('fetch', fetcher);
      render(<App />);
      await screen.findByRole('heading', { name: heading });
      expect(screen.getByRole('alert')).not.toHaveTextContent(
        /FastAPI|backend|Docker|8000|PostgreSQL/i,
      );
      await userEvent.click(screen.getByRole('button', { name: retry }));
      await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
      await screen.findByRole('heading', {
        level: 1,
        name: locale === 'fr' ? 'Collision à l’avant du véhicule' : 'Front collision',
      });
    },
  );
});
