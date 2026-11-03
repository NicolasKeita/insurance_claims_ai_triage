import { proxyReviewRequest } from '../../../../backendProxy';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(request: Request): Promise<Response> {
  return proxyReviewRequest(request);
}
