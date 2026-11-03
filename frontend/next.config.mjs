import { prepareFrontend } from './tooling/prepare.mjs';

const runtime = process.env.CLAIMS_REVIEW_RUNTIME || 'api';
const basePath = process.env.CLAIMS_REVIEW_BASE_PATH || '';
if (!['api', 'static', 'docker'].includes(runtime)) throw new Error('Unsupported review runtime');
if (basePath && !/^\/(?:[A-Za-z0-9_-]+\/)*[A-Za-z0-9_-]+$/.test(basePath)) {
  throw new Error('CLAIMS_REVIEW_BASE_PATH must be a local path without a trailing slash');
}
prepareFrontend({ runtime });

/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(runtime === 'static' ? { output: 'export' } : {}),
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
  pageExtensions: runtime === 'static' ? ['tsx', 'ts'] : ['api.ts', 'tsx', 'ts'],
  env: {
    NEXT_PUBLIC_REVIEW_RUNTIME: runtime,
    NEXT_PUBLIC_REVIEW_BASE_PATH: basePath,
  },
};

export default nextConfig;
