import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { parse } from 'smol-toml';

function placeholders(message) {
  return [...new Set(message.match(/\{[A-Za-z_][A-Za-z_0-9]*\}/g) || [])].sort().join(',');
}

export function compileCatalogues(root) {
  const files = readdirSync(root);
  const complete = ['fr', 'en'].map((locale) => files.includes(`${locale}.toml`));
  if (complete.some(Boolean) && !complete.every(Boolean)) {
    throw new Error('Both fr.toml and en.toml must exist together');
  }
  const catalogues = {};
  for (const locale of ['fr', 'en']) {
    const sources = complete.every(Boolean)
      ? [`${locale}.toml`]
      : files.filter((name) => name.endsWith(`.${locale}.toml`)).sort();
    if (!sources.length) throw new Error(`No ${locale} TOML catalogue found in ${root}`);
    const merged = {};
    for (const name of sources) {
      const payload = parse(readFileSync(path.join(root, name), 'utf8'));
      if (
        Object.keys(payload).join(',') !== 'messages' ||
        !payload.messages ||
        typeof payload.messages !== 'object' ||
        Array.isArray(payload.messages) ||
        !Object.values(payload.messages).every((message) => typeof message === 'string')
      ) {
        throw new Error(`${name}: expected a single [messages] table of strings`);
      }
      Object.assign(merged, payload.messages);
    }
    catalogues[locale] = merged;
  }
  if (
    Object.keys(catalogues.fr).sort().join('\n') !== Object.keys(catalogues.en).sort().join('\n')
  ) {
    throw new Error('Translation catalogue keys differ between FR and EN');
  }
  for (const key of Object.keys(catalogues.en)) {
    if (placeholders(catalogues.fr[key]) !== placeholders(catalogues.en[key])) {
      throw new Error(`Interpolation placeholders differ for ${key}`);
    }
  }
  return catalogues;
}
