export type Theme = 'dark' | 'light';

export const THEME_STORAGE_KEY = 'claims-review-theme';

export function preferredTheme(): Theme {
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY) === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

export function currentTheme(): Theme {
  const theme = document.documentElement.dataset.theme;
  return theme === 'light' || theme === 'dark' ? theme : preferredTheme();
}

export function applyTheme(theme: Theme, persist = true): void {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute('content', theme === 'dark' ? '#10181d' : '#f5f7f7');
  if (persist) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // The current session remains usable when browser storage is unavailable.
    }
  }
}

export function initializeTheme(): void {
  applyTheme(preferredTheme(), false);
}
