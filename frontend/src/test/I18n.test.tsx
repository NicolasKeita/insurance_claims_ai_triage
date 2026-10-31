import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  I18nProvider,
  LOCALE_STORAGE_KEY,
  currentLocale,
  initializeI18n,
  preferredLocale,
  setLocale,
  translateText,
  useI18n,
} from '../i18n';
import { date, decimal, money, percent } from '../presentation';

function storage(): Storage {
  const values = new Map<string, string>();
  return {
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
}

function LocaleProbe() {
  const { locale, t, setLocale: selectLocale } = useI18n();
  return (
    <>
      <output aria-label="Current locale">{locale}</output>
      <p>{t('Available')}</p>
      <p>{t('Source: {source}', { source: 'claim:E2E-VISUAL-0001' })}</p>
      <output aria-label="Formatted date">{date('2026-01-20')}</output>
      <output aria-label="Formatted amount">{money('1240.50', 'EUR')}</output>
      <output aria-label="Formatted decimal">{decimal(1234.56)}</output>
      <output aria-label="Formatted percent">{percent(0.84)}</output>
      <button onClick={() => selectLocale('fr')}>FR</button>
      <button onClick={() => selectLocale('en')}>EN</button>
    </>
  );
}

beforeEach(() => {
  vi.stubGlobal('localStorage', storage());
  initializeI18n();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const plainSpaces = (value: string | null) => value?.replace(/[\u00a0\u202f]/g, ' ');

describe('Local language preference', () => {
  it.each([null, 'unsupported'])(
    'starts in French without a supported saved preference (%s)',
    (saved) => {
      if (saved) window.localStorage.setItem(LOCALE_STORAGE_KEY, saved);
      initializeI18n();
      render(<LocaleProbe />);
      expect(preferredLocale()).toBe('fr');
      expect(currentLocale()).toBe('fr');
      expect(document.documentElement).toHaveAttribute('lang', 'fr');
      expect(screen.getByText('Disponible')).toBeVisible();
      expect(screen.getByText('Source : claim:E2E-VISUAL-0001')).toBeVisible();
    },
  );

  it('selects either language, persists it, and restores it on a new application mount', async () => {
    const first = render(<LocaleProbe />);
    await userEvent.click(screen.getByRole('button', { name: 'EN' }));
    expect(screen.getByText('Available')).toBeVisible();
    expect(document.documentElement).toHaveAttribute('lang', 'en');
    expect(window.localStorage.getItem(LOCALE_STORAGE_KEY)).toBe('en');
    first.unmount();
    setLocale('fr', false);
    render(
      <I18nProvider>
        <LocaleProbe />
      </I18nProvider>,
    );
    expect(screen.getByLabelText('Current locale')).toHaveTextContent('en');
    await userEvent.click(screen.getByRole('button', { name: 'FR' }));
    expect(screen.getByText('Disponible')).toBeVisible();
    expect(window.localStorage.getItem(LOCALE_STORAGE_KEY)).toBe('fr');
    expect(document.documentElement).toHaveAttribute('lang', 'fr');
  });

  it('keeps selection usable when browser storage cannot be read or written', async () => {
    vi.spyOn(window.localStorage, 'getItem').mockImplementation(() => {
      throw new DOMException('Blocked', 'SecurityError');
    });
    vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new DOMException('Full', 'QuotaExceededError');
    });
    expect(initializeI18n).not.toThrow();
    render(<LocaleProbe />);
    expect(screen.getByText('Disponible')).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'EN' }));
    expect(screen.getByText('Available')).toBeVisible();
    expect(document.documentElement).toHaveAttribute('lang', 'en');
  });

  it('formats dates, money and percentages for the selected locale', async () => {
    render(<LocaleProbe />);
    expect(screen.getByLabelText('Formatted date')).toHaveTextContent('20 janv. 2026');
    expect(plainSpaces(screen.getByLabelText('Formatted amount').textContent)).toBe('1 240,50 €');
    expect(plainSpaces(screen.getByLabelText('Formatted decimal').textContent)).toBe('1 234,56');
    expect(plainSpaces(screen.getByLabelText('Formatted percent').textContent)).toBe('84 %');
    await userEvent.click(screen.getByRole('button', { name: 'EN' }));
    expect(screen.getByLabelText('Formatted date')).toHaveTextContent('20 Jan 2026');
    expect(screen.getByLabelText('Formatted amount')).toHaveTextContent('€1,240.50');
    expect(screen.getByLabelText('Formatted decimal')).toHaveTextContent('1,234.56');
    expect(screen.getByLabelText('Formatted percent')).toHaveTextContent('84%');
  });

  it('translates recorded narratives without changing evidence references or unknown source text', () => {
    const source =
      'The claim is for front bumper damage, which is covered under the AUTO_PREMIUM policy (evidence_id: knowledge:2322cae9-b08a-5fda-ae74-34093c5266e9). The deductible is EUR 300 (evidence_id: knowledge:c01cd9c9-d225-5382-8b68-00a997b843fe).';
    const translated = translateText(source, 'fr');
    expect(translated).toContain('La franchise est de 300 EUR');
    expect(translated).toContain('knowledge:2322cae9-b08a-5fda-ae74-34093c5266e9');
    expect(translated).toContain('knowledge:c01cd9c9-d225-5382-8b68-00a997b843fe');
    expect(translateText(source, 'en')).toBe(source);
    expect(translateText('Unknown retained source text', 'fr')).toBe(
      'Unknown retained source text',
    );
    for (const sourceKey of ['constructor', 'toString', '__proto__']) {
      expect(translateText(sourceKey, 'fr')).toBe(sourceKey);
    }
    expect(
      translateText(
        'sha256:ec562cf22561ed00661c40d046351043f2cc0f503af98cb8ebbf561e993b5a1c',
        'fr',
      ),
    ).toBe('sha256:ec562cf22561ed00661c40d046351043f2cc0f503af98cb8ebbf561e993b5a1c');
  });

  it('preserves the policy exclusion and limit when showing its curated translation', () => {
    const source =
      'Intentional damage, mechanical wear, and damage occurring during an organized motor race are excluded. The demo limit for eligible collision repair is EUR 12,000 per incident before applying the deductible. Do not infer coverage for events that this document does not describe.';
    const translated = translateText(source, 'fr');
    expect(translated).toContain('sont exclus');
    expect(translated).toContain('12 000 EUR par sinistre');
    expect(translated).toContain('avant application de la franchise');
    expect(translated).toContain('Ne déduisez aucune couverture');
    expect(translateText(source, 'en')).toBe(source);
  });
});
