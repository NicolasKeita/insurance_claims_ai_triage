import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { proxyReviewRequest } from '../../backendProxy';

const fetchUpstream = vi.fn<typeof fetch>();

beforeEach(() => {
  fetchUpstream.mockReset();
  vi.stubGlobal('fetch', fetchUpstream);
});
afterEach(() => vi.unstubAllGlobals());

function request(path = '/v1/claims/CASE-001/review', init?: RequestInit): Request {
  return new Request(`https://portfolio.example${path}`, init);
}

describe('Next.js review API bridge', () => {
  it('uses the fixed local backend and preserves the requested analysis and cancellation signal', async () => {
    const incoming = request('/v1/claims/CASE-001/review?analysis=saved%20run', {
      headers: { Accept: 'application/json', Cookie: 'unrelated-portfolio-cookie=private' },
    });
    fetchUpstream.mockResolvedValue(Response.json({ claim: { claim_id: 'CASE-001' } }));
    const response = await proxyReviewRequest(incoming);
    const [url, options] = fetchUpstream.mock.calls[0];
    expect(url).toBe('http://127.0.0.1:8000/v1/claims/CASE-001/review?analysis=saved%20run');
    expect(options).toMatchObject({
      method: 'GET',
      signal: incoming.signal,
      cache: 'no-store',
      redirect: 'manual',
    });
    const headers = new Headers(options?.headers);
    expect(headers.get('accept')).toBe('application/json');
    expect(headers.get('accept-encoding')).toBe('identity');
    expect(headers.has('cookie')).toBe(false);
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(await response.json()).toEqual({ claim: { claim_id: 'CASE-001' } });
  });

  it('streams partial PDF responses with file, range and caching headers intact', async () => {
    const pdf = new TextEncoder().encode('%PDF-1.4');
    const upstream = new Response(pdf, {
      status: 206,
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'inline; filename="claim.pdf"',
        'Content-Length': String(pdf.byteLength),
        'Content-Range': 'bytes 0-7/1500',
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'private, max-age=60',
        'X-Content-Type-Options': 'nosniff',
        Connection: 'keep-alive, X-Connection-Only',
        'X-Connection-Only': 'discard',
      },
    });
    fetchUpstream.mockResolvedValue(upstream);
    const response = await proxyReviewRequest(
      request('/v1/claims/CASE-001/review/documents/claim.pdf', {
        headers: { Range: 'bytes=0-7', 'If-Range': 'recorded-etag' },
      }),
    );
    const headers = new Headers(fetchUpstream.mock.calls[0][1]?.headers);
    expect(headers.get('range')).toBe('bytes=0-7');
    expect(headers.get('if-range')).toBe('recorded-etag');
    expect(upstream.bodyUsed).toBe(false);
    expect(response.body).toBe(upstream.body);
    expect(response.status).toBe(206);
    expect(response.headers.get('content-type')).toBe('application/pdf');
    expect(response.headers.get('content-disposition')).toContain('claim.pdf');
    expect(response.headers.get('content-length')).toBe(String(pdf.byteLength));
    expect(response.headers.get('content-range')).toBe('bytes 0-7/1500');
    expect(response.headers.get('accept-ranges')).toBe('bytes');
    expect(response.headers.get('cache-control')).toBe('private, max-age=60');
    expect(response.headers.get('x-content-type-options')).toBe('nosniff');
    expect(response.headers.has('connection')).toBe(false);
    expect(response.headers.has('x-connection-only')).toBe(false);
    expect(Array.from(new Uint8Array(await response.arrayBuffer()))).toEqual(Array.from(pdf));
  });

  it.each([404, 503])('preserves an upstream HTTP %s and its response body', async (status) => {
    fetchUpstream.mockResolvedValue(
      Response.json(
        { detail: 'Recorded claim unavailable' },
        { status, headers: { 'Retry-After': '15' } },
      ),
    );
    const response = await proxyReviewRequest(request());
    expect(response.status).toBe(status);
    expect(response.headers.get('retry-after')).toBe('15');
    expect(await response.json()).toEqual({ detail: 'Recorded claim unavailable' });
  });

  it('returns a stable transport failure without disclosing private connection details', async () => {
    fetchUpstream.mockRejectedValue(new Error('ECONNREFUSED private transport details'));
    const response = await proxyReviewRequest(request());
    expect(response.status).toBe(502);
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(await response.json()).toEqual({ error: { code: 'BACKEND_UNREACHABLE' } });
  });

  it('propagates cancellation instead of reporting an unavailable backend', async () => {
    const controller = new AbortController();
    fetchUpstream.mockImplementation(
      (_url, options) =>
        new Promise((_resolve, reject) => {
          options?.signal?.addEventListener('abort', () => reject(options.signal?.reason), {
            once: true,
          });
        }),
    );
    const incoming = request(undefined, { signal: controller.signal });
    const pending = proxyReviewRequest(incoming);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetchUpstream.mock.calls[0][1]?.signal).toBe(incoming.signal);
  });

  it('does not contact the backend for a request cancelled before routing', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      proxyReviewRequest(request(undefined, { signal: controller.signal })),
    ).rejects.toMatchObject({
      name: 'AbortError',
    });
    expect(fetchUpstream).not.toHaveBeenCalled();
  });

  it('does not follow backend redirects or buffer their response body', async () => {
    fetchUpstream.mockResolvedValue(
      new Response(null, { status: 307, headers: { Location: '/v1/review/demo-claims' } }),
    );
    const response = await proxyReviewRequest(request());
    expect(response.status).toBe(307);
    expect(response.headers.get('location')).toBe('/v1/review/demo-claims');
    expect(fetchUpstream.mock.calls[0][1]?.redirect).toBe('manual');
  });

  it('removes compressed transfer metadata after Fetch has decoded the body', async () => {
    fetchUpstream.mockResolvedValue(
      new Response('decoded photo bytes', {
        headers: {
          'Content-Type': 'image/jpeg',
          'Content-Encoding': 'gzip',
          'Content-Length': '12',
        },
      }),
    );
    const response = await proxyReviewRequest(
      request('/v1/claims/CASE-001/review/images/photo.jpg'),
    );
    expect(response.headers.get('content-type')).toBe('image/jpeg');
    expect(response.headers.has('content-encoding')).toBe(false);
    expect(response.headers.has('content-length')).toBe(false);
    expect(await response.text()).toBe('decoded photo bytes');
  });

  it('does not forward unrelated paths or mutation methods', async () => {
    expect((await proxyReviewRequest(request('/private'))).status).toBe(404);
    const mutation = await proxyReviewRequest(
      request(undefined, { method: 'POST', body: 'discard' }),
    );
    expect(mutation.status).toBe(405);
    expect(mutation.headers.get('allow')).toBe('GET, HEAD');
    expect(fetchUpstream).not.toHaveBeenCalled();
  });
});
