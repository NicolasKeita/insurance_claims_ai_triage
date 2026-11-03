import { useEffect, useMemo, useSyncExternalStore } from 'react';
import type { ReactNode } from 'react';
import translations from './generated/locales.json';

const catalogues: Record<'fr' | 'en', Record<string, string>> = translations;

export type Locale = 'fr' | 'en';
export const LOCALE_STORAGE_KEY = 'claims-review-locale';
const listeners = new Set<() => void>();

export function preferredLocale(): Locale {
  try {
    const linkedLocale = new URLSearchParams(window.location.search).get('lang');
    if (linkedLocale === 'fr' || linkedLocale === 'en') return linkedLocale;
    return window.localStorage.getItem(LOCALE_STORAGE_KEY) === 'en' ? 'en' : 'fr';
  } catch {
    return 'fr';
  }
}

let activeLocale: Locale = 'fr';

export function currentLocale(): Locale {
  return activeLocale;
}

export function setLocale(locale: Locale, persist = true): void {
  activeLocale = locale === 'en' ? 'en' : 'fr';
  if (typeof document !== 'undefined') document.documentElement.lang = activeLocale;
  if (persist) {
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      if (url.searchParams.has('lang')) {
        url.searchParams.set('lang', activeLocale);
        window.history.replaceState(null, '', url);
      }
    }
    try {
      window.localStorage.setItem(LOCALE_STORAGE_KEY, activeLocale);
    } catch {
      // Language selection remains usable when browser storage is unavailable.
    }
  }
  listeners.forEach((listener) => listener());
}

export function initializeI18n(): void {
  setLocale(preferredLocale(), false);
}

export function translateText(text: string, locale: Locale = currentLocale()): string {
  return Object.hasOwn(catalogues[locale], text) ? catalogues[locale][text] : text;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useI18n() {
  const locale = useSyncExternalStore(subscribe, currentLocale, () => 'fr' as const);
  return useMemo(
    () => ({
      locale,
      setLocale,
      translate: (text: string) => translateText(text, locale),
      t: (english: string, values: Record<string, string | number> = {}) =>
        translateText(english, locale).replace(
          /\{([A-Za-z_][A-Za-z_0-9]*)\}/g,
          (token, key: string) => (Object.hasOwn(values, key) ? String(values[key]) : token),
        ),
    }),
    [locale],
  );
}

export function I18nProvider({
  children,
  initialLocale,
}: {
  children: ReactNode;
  initialLocale?: Locale;
}) {
  useEffect(() => {
    setLocale(initialLocale ?? preferredLocale(), false);
  }, [initialLocale]);
  return <>{children}</>;
}
