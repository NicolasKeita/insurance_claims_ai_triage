import type { ProxyOptions } from 'vite';

/** Preserve upstream HTTP errors; identify a transport failure before Vite's generic 500. */
export function reviewBackendProxy(): ProxyOptions {
  return {
    target: 'http://127.0.0.1:8000',
    configure(proxy) {
      proxy.on('error', (_error, _request, response) => {
        if (
          !('writeHead' in response) ||
          response.headersSent ||
          response.writableEnded ||
          response.destroyed
        ) {
          return;
        }
        response.writeHead(502, {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store',
        });
        response.end(JSON.stringify({ error: { code: 'BACKEND_UNREACHABLE' } }));
      });
    },
  };
}
