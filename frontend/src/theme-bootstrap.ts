/** Applied before hydration so a saved light preference does not flash a dark canvas. */
export const themeBootstrap = `try {
  if (localStorage.getItem('claims-review-theme') === 'light') {
    document.documentElement.dataset.theme = 'light';
  }
} catch { /* Keep the dark default when storage is unavailable. */ }`;
