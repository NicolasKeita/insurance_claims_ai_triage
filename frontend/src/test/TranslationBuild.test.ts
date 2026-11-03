import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { compileCatalogues } from '../../tooling/compile-locales.mjs';

const folders: string[] = [];
afterEach(() => {
  for (const folder of folders.splice(0)) rmSync(folder, { recursive: true, force: true });
});

function fixture(fr: string, en: string) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'claims-locales-'));
  folders.push(root);
  writeFileSync(path.join(root, 'fr.toml'), `[messages]\n${fr}\n`);
  writeFileSync(path.join(root, 'en.toml'), `[messages]\n${en}\n`);
  return root;
}

describe('Frontend translation build without Python', () => {
  it('parses the real bilingual catalogues with the Node TOML parser', () => {
    const messages = compileCatalogues(path.resolve('locales'));
    expect(messages.fr['Back to portfolio']).toBe('Retour au portfolio');
    expect(messages.en['Back to portfolio']).toBe('Back to portfolio');
    expect(Object.keys(messages.fr).length).toBeGreaterThan(100);
  });

  it('rejects a missing translation instead of shipping partial FR/EN content', () => {
    expect(() => compileCatalogues(fixture('"Hello" = "Bonjour"', '"Other" = "Other"'))).toThrow(
      /keys differ/,
    );
  });

  it('rejects lost interpolation values', () => {
    expect(() =>
      compileCatalogues(fixture('"Count" = "Dossiers"', '"Count" = "{count} cases"')),
    ).toThrow(/placeholders differ/);
  });
});
