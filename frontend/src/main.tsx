import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';
import './review-layout.css';
import './demo-flow.css';
import { initializeTheme } from './theme';

initializeTheme();
createRoot(document.getElementById('root')!).render(<App />);
