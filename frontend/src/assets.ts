export function isStaticDemo(): boolean {
  return process.env.NEXT_PUBLIC_REVIEW_RUNTIME === 'static';
}

export function appBaseUrl(): string {
  const base = process.env.NEXT_PUBLIC_REVIEW_BASE_PATH || '';
  if (base && !/^\/(?:[A-Za-z0-9_-]+\/)*[A-Za-z0-9_-]+\/?$/.test(base)) return '/';
  return base.endsWith('/') ? base : `${base}/`;
}

function safeSegments(path: string): string[] | null {
  try {
    const parts = path.split('/').map((part) => decodeURIComponent(part));
    if (
      parts.some(
        (part) =>
          !part ||
          part === '.' ||
          part === '..' ||
          /[\\/%?#]/.test(part) ||
          [...part].some(
            (character) => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127,
          ),
      )
    )
      return null;
    return parts;
  } catch {
    return null;
  }
}

export function demoFileUrl(path: string): string | null {
  const base = appBaseUrl();
  const relative = path.startsWith(base) ? path.slice(base.length) : path;
  const parts = safeSegments(relative);
  if (!parts || parts[0] !== 'demo') return null;
  return `${base}${parts.map(encodeURIComponent).join('/')}`;
}

export function reviewAssetUrl(
  value: string | null | undefined,
  kind: 'documents' | 'images',
): string | null {
  if (!value) return null;
  if (isStaticDemo()) {
    const url = demoFileUrl(value);
    if (!url) return null;
    const parts = url.slice(appBaseUrl().length).split('/');
    if (parts.length !== 4 || parts[1] !== kind) return null;
    const extension = kind === 'documents' ? /\.pdf$/i : /\.(png|jpe?g|webp|gif|bmp|avif)$/i;
    return extension.test(parts[3]) ? url : null;
  }
  const [path, query] = value.split('?');
  if (!path.startsWith('/v1/') || !safeSegments(path.slice(1)) || value.includes('#')) return null;
  if (query && [...new URLSearchParams(query).keys()].some((key) => key !== 'analysis'))
    return null;
  return value;
}
