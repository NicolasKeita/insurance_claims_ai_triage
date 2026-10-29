import { useState } from 'react';
import { applyTheme, currentTheme } from '../theme';

export function ThemeToggle() {
  const [theme, setTheme] = useState(currentTheme);
  const nextTheme = theme === 'dark' ? 'light' : 'dark';
  const label = `Use ${nextTheme} mode`;

  function toggle() {
    applyTheme(nextTheme);
    setTheme(nextTheme);
  }

  return (
    <button
      className="theme-toggle"
      type="button"
      aria-label={label}
      title={label}
      onClick={toggle}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        aria-hidden="true"
      >
        {theme === 'dark' ? (
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
          </>
        ) : (
          <path d="M20 15.5A8.5 8.5 0 0 1 8.5 4 8.5 8.5 0 1 0 20 15.5Z" />
        )}
      </svg>
      <span>{nextTheme === 'light' ? 'Light mode' : 'Dark mode'}</span>
    </button>
  );
}
