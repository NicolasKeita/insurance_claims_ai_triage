export type RequestFailureKind =
  'backend-unavailable' | 'claim-not-found' | 'service-unavailable' | 'request-failed';

type Resource = 'claim' | 'catalog';

export class RequestFailure extends Error {
  constructor(
    public readonly kind: RequestFailureKind,
    public readonly status?: number,
    public readonly resource: Resource = 'claim',
  ) {
    super(kind);
    this.name = 'RequestFailure';
  }
}

export function requestFailure(error: unknown): RequestFailure {
  return error instanceof RequestFailure ? error : new RequestFailure('request-failed');
}

export function isAbortError(error: unknown): boolean {
  return (
    typeof error === 'object' && error !== null && 'name' in error && error.name === 'AbortError'
  );
}

function isBackendSentinel(payload: unknown): boolean {
  if (typeof payload !== 'object' || payload === null || !('error' in payload)) return false;
  const error = payload.error;
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    error.code === 'BACKEND_UNREACHABLE'
  );
}

export async function fetchJson<T>(
  url: string,
  { signal, resource }: { signal?: AbortSignal; resource: Resource },
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { signal, headers: { Accept: 'application/json' } });
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) throw error;
    throw new RequestFailure('backend-unavailable', undefined, resource);
  }
  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    if (isBackendSentinel(payload) || response.status === 502 || response.status === 504) {
      throw new RequestFailure('backend-unavailable', response.status, resource);
    }
    if (response.status === 404 && resource === 'claim') {
      throw new RequestFailure('claim-not-found', response.status, resource);
    }
    if (response.status === 503) {
      throw new RequestFailure('service-unavailable', response.status, resource);
    }
    throw new RequestFailure('request-failed', response.status, resource);
  }
  try {
    return (await response.json()) as T;
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) throw error;
    throw new RequestFailure('request-failed', response.status, resource);
  }
}
