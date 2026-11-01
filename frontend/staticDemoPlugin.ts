import { readFileSync, lstatSync } from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'vite';

/** Fail at preparation time instead of shipping a console without its dossiers. */
export function reviewStaticDemo(): Plugin {
  let publicRoot = '';
  return {
    name: 'claims-static-demo',
    configResolved(config) {
      publicRoot = config.publicDir;
    },
    buildStart() {
      const demoRoot = path.join(publicRoot, 'demo');
      try {
        const manifest = JSON.parse(readFileSync(path.join(demoRoot, 'manifest.json'), 'utf8'));
        const catalog = JSON.parse(readFileSync(path.join(demoRoot, 'catalog.json'), 'utf8'));
        if (
          [demoRoot, path.join(demoRoot, 'reviews')].some((folder) =>
            lstatSync(folder).isSymbolicLink(),
          )
        ) {
          throw new Error('Static demo directories must be ordinary directories.');
        }
        if (
          manifest.format !== 'claims_static_demo_v1' ||
          !Array.isArray(manifest.claims) ||
          !manifest.claims.length ||
          !Array.isArray(catalog) ||
          !catalog.length
        ) {
          throw new Error('The static catalogue is empty or unsupported.');
        }
        for (const entry of manifest.claims) {
          if (typeof entry.review !== 'string' || !/^reviews\/[\w.-]+\.json$/.test(entry.review)) {
            throw new Error('The static manifest contains an invalid review path.');
          }
          const reviewPath = path.join(demoRoot, entry.review);
          if (!lstatSync(reviewPath).isFile()) throw new Error('A prepared review is missing.');
          const review = JSON.parse(readFileSync(reviewPath, 'utf8'));
          if (review.claim?.claim_id !== entry.claim_id || review.mode !== 'demo') {
            throw new Error('A prepared review differs from its demo identity.');
          }
        }
      } catch (error) {
        throw new Error(
          'Static demo files are unavailable. Restore frontend/public/demo or run ' +
            'scripts/export_static_demo.py with an existing analysis collection.',
          { cause: error },
        );
      }
    },
  };
}
