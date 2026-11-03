import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { appBaseUrl, reviewAssetUrl } from '../assets';

beforeEach(() => {
  vi.stubEnv('NEXT_PUBLIC_REVIEW_RUNTIME', 'static');
  vi.stubEnv('NEXT_PUBLIC_REVIEW_BASE_PATH', '/portfolio/');
});
afterEach(() => vi.unstubAllEnvs());

describe('Local static evidence URLs', () => {
  it('resolves copied PDFs and photographs under the configured hosting base', () => {
    expect(appBaseUrl()).toBe('/portfolio/');
    expect(reviewAssetUrl('demo/documents/CLAIM_ANALYSIS/claim_form.pdf', 'documents')).toBe(
      '/portfolio/demo/documents/CLAIM_ANALYSIS/claim_form.pdf',
    );
    expect(
      reviewAssetUrl('/portfolio/demo/images/CLAIM_ANALYSIS/source%20photo.jpg', 'images'),
    ).toBe('/portfolio/demo/images/CLAIM_ANALYSIS/source%20photo.jpg');
  });

  it.each([
    'https://evil.example/demo/documents/A/claim.pdf',
    '//evil.example/demo/documents/A/claim.pdf',
    'javascript:alert(1)',
    '/elsewhere/demo/documents/A/claim.pdf',
    'demo/documents/../claim.pdf',
    'demo/documents/%2e%2e/claim.pdf',
    'demo/documents/%252e%252e/claim.pdf',
    'demo/documents/A%2f..%2fB/claim.pdf',
    'demo/documents/A%5c..%5cB/claim.pdf',
    'demo/documents/A/claim.pdf?next=https://evil.example',
    'demo/documents/A/claim.pdf#fragment',
    'demo/documents/A/%00claim.pdf',
    'demo/documents/A/%GG.pdf',
    'demo/documents/A/claim.html',
    'demo/images/A/claim.pdf',
    '/v1/claims/CLAIM/review/documents/claim.pdf',
  ])('does not expose an unsafe or non-document URL: %s', (value) => {
    expect(reviewAssetUrl(value, 'documents')).toBeNull();
  });

  it('rejects executable image formats and cross-category paths', () => {
    expect(reviewAssetUrl('demo/images/A/source.svg', 'images')).toBeNull();
    expect(reviewAssetUrl('demo/documents/A/source.png', 'images')).toBeNull();
  });

  it('retains registered API asset links with analysis selection in API mode', () => {
    vi.stubEnv('NEXT_PUBLIC_REVIEW_RUNTIME', 'api');
    const url = '/v1/claims/CLAIM/review/documents/claim.pdf?analysis=ANALYSIS-01';
    expect(reviewAssetUrl(url, 'documents')).toBe(url);
    expect(reviewAssetUrl('/v1/../../outside/claim.pdf', 'documents')).toBeNull();
    expect(reviewAssetUrl('/v1/%252e%252e/claim.pdf', 'documents')).toBeNull();
    expect(reviewAssetUrl('demo/documents/A/claim.pdf', 'documents')).toBeNull();
  });
});
