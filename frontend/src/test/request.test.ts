import { describe, expect, it, vi } from 'vitest';
import { fetchJson } from '../request';

describe('request cancellation', () => {
  it('preserves AbortError instead of reporting an unavailable backend', async () => {
    const aborted = new DOMException('Request cancelled', 'AbortError');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(aborted));
    await expect(fetchJson('/v1/review/demo-claims', { resource: 'catalog' })).rejects.toBe(
      aborted,
    );
  });

  it('preserves cancellation while a successful response body is being read', async () => {
    const aborted = new DOMException('Body read cancelled', 'AbortError');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => {
          throw aborted;
        },
      }),
    );
    await expect(fetchJson('/v1/claims/CLAIM/review', { resource: 'claim' })).rejects.toBe(aborted);
  });
});
