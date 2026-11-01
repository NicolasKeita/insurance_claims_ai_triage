import { demoFileUrl, isStaticDemo, reviewAssetUrl } from './assets';
import { fetchJson, isAbortError, RequestFailure } from './request';
import type { DemoClaim, ReviewView } from './types';

interface StaticEntry {
  claim_id: string;
  analysis_id: string;
  review: string;
}

interface StaticManifest {
  format: string;
  claims: StaticEntry[];
}

async function staticJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const url = demoFileUrl(`demo/${path}`);
  if (!url) throw new RequestFailure('request-failed');
  try {
    return await fetchJson<T>(url, { signal, resource: 'catalog' });
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) throw error;
    throw new RequestFailure(
      'request-failed',
      error instanceof RequestFailure ? error.status : undefined,
    );
  }
}

export async function getDemoClaims(signal?: AbortSignal): Promise<DemoClaim[]> {
  if (!isStaticDemo()) return fetchJson('/v1/review/demo-claims', { signal, resource: 'catalog' });
  const claims = await staticJson<DemoClaim[]>('catalog.json', signal);
  if (
    !Array.isArray(claims) ||
    !claims.every(
      (claim) =>
        claim &&
        typeof claim.claim_id === 'string' &&
        typeof claim.title === 'string' &&
        typeof claim.description === 'string',
    )
  )
    throw new RequestFailure('request-failed', undefined, 'catalog');
  return claims;
}

function staticAssets(review: ReviewView): ReviewView {
  return {
    ...review,
    documents: {
      ...review.documents,
      data: review.documents.data
        ? {
            ...review.documents.data,
            items: review.documents.data.items.map((item) => ({
              ...item,
              document_url: reviewAssetUrl(item.document_url, 'documents'),
            })),
          }
        : null,
    },
    vision: {
      ...review.vision,
      data: review.vision.data
        ? {
            ...review.vision.data,
            images: review.vision.data.images.map((image) => ({
              ...image,
              image_url: reviewAssetUrl(image.image_url, 'images'),
            })),
          }
        : null,
    },
  };
}

export async function getClaimReview(
  id: string,
  analysis: string | null,
  signal?: AbortSignal,
): Promise<ReviewView> {
  if (!isStaticDemo()) {
    const query = analysis ? `?analysis=${encodeURIComponent(analysis)}` : '';
    return fetchJson(`/v1/claims/${encodeURIComponent(id)}/review${query}`, {
      signal,
      resource: 'claim',
    });
  }
  const manifest = await staticJson<StaticManifest>('manifest.json', signal);
  if (
    manifest?.format !== 'claims_static_demo_v1' ||
    !Array.isArray(manifest.claims) ||
    !manifest.claims.every(
      (entry) =>
        entry &&
        typeof entry.claim_id === 'string' &&
        typeof entry.analysis_id === 'string' &&
        typeof entry.review === 'string',
    )
  ) {
    throw new RequestFailure('request-failed');
  }
  const entry = manifest.claims
    .filter((item) => item.claim_id === id && (!analysis || item.analysis_id === analysis))
    .at(-1);
  if (!entry) throw new RequestFailure('claim-not-found', 404);
  if (!/^reviews\/[A-Za-z0-9][A-Za-z0-9_.-]*\.json$/.test(entry.review)) {
    throw new RequestFailure('request-failed');
  }
  const review = await staticJson<ReviewView>(entry.review, signal);
  const sections = [
    'documents',
    'consistency',
    'triage',
    'anomaly',
    'history',
    'investigation',
    'vision',
    'visual_consistency',
    'similar_claims',
    'knowledge',
    'agent',
  ] as const;
  if (
    review?.claim?.claim_id !== id ||
    !sections.every((key) => review[key] && typeof review[key].status === 'string') ||
    !Array.isArray(review.evidence)
  ) {
    throw new RequestFailure('request-failed');
  }
  return staticAssets(review);
}
