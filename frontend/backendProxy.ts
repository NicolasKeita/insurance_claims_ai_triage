const backendOrigin = 'http://127.0.0.1:8000';
const requestHeaders = ['accept', 'range', 'if-range', 'if-none-match', 'if-modified-since'];
const hopHeaders = [
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
];

/** A read-only local API bridge; files retain their streaming body and range headers. */
export async function proxyReviewRequest(request: Request): Promise<Response> {
  const source = new URL(request.url);
  if (!source.pathname.startsWith('/v1/')) return new Response(null, { status: 404 });
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return new Response(null, { status: 405, headers: { Allow: 'GET, HEAD' } });
  }
  request.signal.throwIfAborted();
  const headers = new Headers({ 'Accept-Encoding': 'identity' });
  for (const name of requestHeaders) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  let upstream: Response;
  try {
    upstream = await fetch(`${backendOrigin}${source.pathname}${source.search}`, {
      method: request.method,
      headers,
      signal: request.signal,
      cache: 'no-store',
      redirect: 'manual',
    });
  } catch (error) {
    if (request.signal.aborted) throw request.signal.reason ?? error;
    if (
      typeof error === 'object' &&
      error !== null &&
      'name' in error &&
      error.name === 'AbortError'
    ) {
      throw error;
    }
    return Response.json(
      { error: { code: 'BACKEND_UNREACHABLE' } },
      { status: 502, headers: { 'Cache-Control': 'no-store' } },
    );
  }
  const responseHeaders = new Headers(upstream.headers);
  const connectionHeaders = upstream.headers.get('connection')?.split(',') ?? [];
  for (const name of [...hopHeaders, ...connectionHeaders.map((value) => value.trim())]) {
    if (name) responseHeaders.delete(name);
  }
  // Fetch decodes encoded response bodies; do not forward their compressed lengths.
  if (responseHeaders.has('content-encoding')) {
    responseHeaders.delete('content-encoding');
    responseHeaders.delete('content-length');
  }
  if (!responseHeaders.has('cache-control')) responseHeaders.set('Cache-Control', 'no-store');
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
}
