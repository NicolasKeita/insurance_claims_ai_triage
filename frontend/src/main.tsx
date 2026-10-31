import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';
import './review-layout.css';
import './demo-flow.css';
import { initializeTheme } from './theme';
import { I18nProvider, initializeI18n } from './i18n';

initializeTheme();
initializeI18n();
createRoot(document.getElementById('root')!).render(
  <I18nProvider>
    <App />
  </I18nProvider>,
);
