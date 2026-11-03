import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getClaimReview, getDemoClaims } from '../reviewData';
import { partialReview } from './fixtures';

const claims = [
  { claim_id: 'DEMO-COHERENT', title: 'Prepared case', description: 'Recorded evidence' },
];
const manifest = {
  format: 'claims_static_demo_v1',
  claims: [
    { claim_id: 'DEMO-COHERENT', analysis_id: 'OLD', review: 'reviews/DEMO_OLD.json' },
    { claim_id: 'DEMO-COHERENT', analysis_id: 'LATEST', review: 'reviews/DEMO_LATEST.json' },
  ],
};

function response(value: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => value };
}

beforeEach(() => {
  vi.stubEnv('NEXT_PUBLIC_REVIEW_RUNTIME', 'static');
  vi.stubEnv('NEXT_PUBLIC_REVIEW_BASE_PATH', '/portfolio/');
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe('Static review data provider', () => {
  it.each([null, ''])(
    'reads the catalogue and latest analysis using only local files (%s)',
    async (analysis) => {
      const review = partialReview();
      review.documents.data!.items[0].document_url = 'demo/documents/DEMO_LATEST/claim_form.pdf';
      const original = JSON.stringify(review);
      const fetcher = vi
        .fn()
        .mockResolvedValueOnce(response(claims))
        .mockResolvedValueOnce(response(manifest))
        .mockResolvedValueOnce(response(review));
      vi.stubGlobal('fetch', fetcher);
      expect(await getDemoClaims()).toEqual(claims);
      const actual = await getClaimReview('DEMO-COHERENT', analysis);
      expect(actual.documents.data!.items[0].document_url).toBe(
        '/portfolio/demo/documents/DEMO_LATEST/claim_form.pdf',
      );
      expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
        '/portfolio/demo/catalog.json',
        '/portfolio/demo/manifest.json',
        '/portfolio/demo/reviews/DEMO_LATEST.json',
      ]);
      expect(JSON.stringify(review)).toBe(original);
    },
  );

  it('honours a particular analysis instead of silently returning latest', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(manifest))
      .mockResolvedValueOnce(response(partialReview()));
    vi.stubGlobal('fetch', fetcher);
    await getClaimReview('DEMO-COHERENT', 'OLD');
    expect(fetcher.mock.calls[1][0]).toBe('/portfolio/demo/reviews/DEMO_OLD.json');
  });

  it.each([
    ['UNKNOWN', null],
    ['DEMO-COHERENT', 'UNKNOWN'],
  ])('reports an unknown claim/analysis as 404: %s / %s', async (id, analysis) => {
    const fetcher = vi.fn().mockResolvedValue(response(manifest));
    vi.stubGlobal('fetch', fetcher);
    await expect(getClaimReview(id!, analysis)).rejects.toMatchObject({
      kind: 'claim-not-found',
      status: 404,
    });
    expect(fetcher).toHaveBeenCalledOnce();
  });

  it.each(['../outside.json', 'reviews/%2e%2e.json', 'https://evil.example/review.json'])(
    'rejects an invalid manifest path before issuing a second request: %s',
    async (review) => {
      const fetcher = vi.fn().mockResolvedValue(
        response({
          ...manifest,
          claims: [{ ...manifest.claims[0], review }],
        }),
      );
      vi.stubGlobal('fetch', fetcher);
      await expect(getClaimReview('DEMO-COHERENT', null)).rejects.toMatchObject({
        kind: 'request-failed',
      });
      expect(fetcher).toHaveBeenCalledOnce();
    },
  );

  it.each([null, {}, { format: 'wrong', claims: [] }])(
    'rejects a malformed manifest: %s',
    async (payload) => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response(payload)));
      await expect(getClaimReview('DEMO-COHERENT', null)).rejects.toMatchObject({
        kind: 'request-failed',
      });
    },
  );

  it('rejects a snapshot for a different claim', async () => {
    const review = partialReview();
    review.claim.claim_id = 'OTHER';
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValueOnce(response(manifest)).mockResolvedValueOnce(response(review)),
    );
    await expect(getClaimReview('DEMO-COHERENT', null)).rejects.toMatchObject({
      kind: 'request-failed',
    });
  });

  it('preserves aborted requests without reporting demo corruption', async () => {
    const aborted = new DOMException('Aborted', 'AbortError');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(aborted));
    await expect(getDemoClaims()).rejects.toBe(aborted);
  });
});
