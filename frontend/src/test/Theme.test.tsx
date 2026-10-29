import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ThemeToggle } from '../components/ThemeToggle';
import { initializeTheme, THEME_STORAGE_KEY } from '../theme';

beforeEach(() => {
  const values = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return values.size;
    },
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      values.set(key, value);
    },
    removeItem: (key) => {
      values.delete(key);
    },
    clear: () => values.clear(),
    key: (index) => Array.from(values.keys())[index] ?? null,
  };
  // Use controlled browser storage instead of Node's experimental Web Storage global.
  vi.stubGlobal('localStorage', storage);
  delete document.documentElement.dataset.theme;
  document.documentElement.style.removeProperty('color-scheme');
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  delete document.documentElement.dataset.theme;
  document.documentElement.style.removeProperty('color-scheme');
});

describe('Console theme preference', () => {
  it('starts dark when no preference has been saved', () => {
    initializeTheme();
    render(<ThemeToggle />);
    expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
    expect(document.documentElement.style.colorScheme).toBe('dark');
    expect(screen.getByRole('button', { name: 'Use light mode' })).toBeEnabled();
  });

  it('switches by keyboard and remembers the choice across a new application mount', async () => {
    initializeTheme();
    const first = render(<ThemeToggle />);
    const user = userEvent.setup();
    await user.tab();
    expect(screen.getByRole('button', { name: 'Use light mode' })).toHaveFocus();
    await user.keyboard('{Enter}');
    expect(document.documentElement).toHaveAttribute('data-theme', 'light');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');
    first.unmount();
    initializeTheme();
    render(<ThemeToggle />);
    await user.click(screen.getByRole('button', { name: 'Use dark mode' }));
    expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
    expect(document.documentElement.style.colorScheme).toBe('dark');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
  });

  it('honours an explicitly saved light preference at startup', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
    initializeTheme();
    render(<ThemeToggle />);
    expect(document.documentElement.style.colorScheme).toBe('light');
    expect(screen.getByRole('button', { name: 'Use dark mode' })).toBeEnabled();
  });

  it('uses the dark default for an invalid stored preference', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'unrecognised');
    initializeTheme();
    expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
  });

  it('keeps the toggle usable when browser storage cannot be read or written', async () => {
    vi.spyOn(window.localStorage, 'getItem').mockImplementation(() => {
      throw new DOMException('Storage blocked', 'SecurityError');
    });
    vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new DOMException('Storage full', 'QuotaExceededError');
    });
    initializeTheme();
    render(<ThemeToggle />);
    await userEvent.click(screen.getByRole('button', { name: 'Use light mode' }));
    expect(document.documentElement).toHaveAttribute('data-theme', 'light');
    expect(screen.getByRole('button', { name: 'Use dark mode' })).toBeEnabled();
  });
});

describe('Theme before the application bundle loads', () => {
  const html = readFileSync('index.html', 'utf8');
  const script = html.match(/<script id="theme-bootstrap">([\s\S]*?)<\/script>/)?.[1];

  it.each(['light', 'dark', null])(
    'agrees with application startup for saved preference %s',
    (saved) => {
      if (saved) window.localStorage.setItem(THEME_STORAGE_KEY, saved);
      document.documentElement.dataset.theme = 'dark';
      expect(script).toBeDefined();
      runInNewContext(script!, { document, localStorage: window.localStorage });
      const prepaintTheme = document.documentElement.dataset.theme;
      initializeTheme();
      expect(document.documentElement.dataset.theme).toBe(prepaintTheme);
      expect(prepaintTheme).toBe(saved === 'light' ? 'light' : 'dark');
    },
  );

  it('does not expose a light canvas when startup storage access fails', () => {
    document.documentElement.dataset.theme = 'dark';
    expect(() =>
      runInNewContext(script!, {
        document,
        localStorage: {
          getItem() {
            throw new Error('Storage unavailable');
          },
        },
      }),
    ).not.toThrow();
    expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
  });
});
