import { lstatSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { compileCatalogues } from './compile-locales.mjs';

export const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function validateStaticDemo(root) {
  const demoRoot = path.join(root, 'public', 'demo');
  const manifest = JSON.parse(readFileSync(path.join(demoRoot, 'manifest.json'), 'utf8'));
  const catalogue = JSON.parse(readFileSync(path.join(demoRoot, 'catalog.json'), 'utf8'));
  if (
    manifest.format !== 'claims_static_demo_v1' ||
    !Array.isArray(manifest.claims) ||
    !manifest.claims.length ||
    !Array.isArray(catalogue) ||
    !catalogue.length
  )
    throw new Error('The static demo catalogue is empty or unsupported');
  if (
    [demoRoot, path.join(demoRoot, 'reviews')].some((folder) => lstatSync(folder).isSymbolicLink())
  ) {
    throw new Error('Static demo directories must be ordinary directories');
  }
  for (const entry of manifest.claims) {
    if (typeof entry.review !== 'string' || !/^reviews\/[\w.-]+\.json$/.test(entry.review)) {
      throw new Error('The static manifest contains an invalid review path');
    }
    const reviewPath = path.join(demoRoot, entry.review);
    const stat = lstatSync(reviewPath);
    if (!stat.isFile() || stat.isSymbolicLink()) throw new Error('A prepared review is missing');
    const review = JSON.parse(readFileSync(reviewPath, 'utf8'));
    if (review.claim?.claim_id !== entry.claim_id || review.mode !== 'demo') {
      throw new Error('A prepared review differs from its demo identity');
    }
    if (!catalogue.some((claim) => claim.claim_id === entry.claim_id)) {
      throw new Error('A prepared review is absent from the catalogue');
    }
  }
}

export function prepareFrontend({
  root = frontendRoot,
  runtime = process.env.CLAIMS_REVIEW_RUNTIME,
} = {}) {
  const catalogues = compileCatalogues(path.join(root, 'locales'));
  const generated = path.join(root, 'src', 'generated');
  mkdirSync(generated, { recursive: true });
  const target = path.join(generated, 'locales.json');
  const content = `${JSON.stringify(catalogues)}\n`;
  let existing;
  try {
    existing = readFileSync(target, 'utf8');
  } catch {
    /* First preparation. */
  }
  if (existing !== content) writeFileSync(target, content);
  if (runtime === 'static') validateStaticDemo(root);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  prepareFrontend();
}
