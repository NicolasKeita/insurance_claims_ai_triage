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
import { date, decimal, money, narrative, percent, rationaleText } from '../presentation';

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
  window.history.replaceState(null, '', '/');
  vi.stubGlobal('localStorage', storage());
  initializeI18n();
});

afterEach(() => {
  window.history.replaceState(null, '', '/');
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const plainSpaces = (value: string | null) => value?.replace(/[\u00a0\u202f]/g, ' ');

describe('Local language preference', () => {
  it('uses the portfolio language link and keeps it consistent after switching languages', async () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'fr');
    window.history.replaceState(null, '', '/insurance-demo/?lang=en&claim=DEMO-1');
    initializeI18n();
    render(<LocaleProbe />);
    expect(screen.getByText('Available')).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'FR' }));
    expect(new URLSearchParams(window.location.search).get('lang')).toBe('fr');
    expect(new URLSearchParams(window.location.search).get('claim')).toBe('DEMO-1');
    expect(preferredLocale()).toBe('fr');
  });

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

  it('shows prose confidence as a concise localized percentage while retaining other numbers', () => {
    const source =
      'The triage assessment recommends FAST_TRACK with a confidence of 0.40971294756240506. Source triage:0.40971294756240506.';
    expect(plainSpaces(narrative(source))).toBe(
      'The triage assessment recommends FAST_TRACK with a confidence of 41 %. Source triage:0.40971294756240506.',
    );
    expect(plainSpaces(narrative('Confiance de 0,40971294756240506.'))).toBe('Confiance de 41 %.');
    setLocale('en', false);
    expect(narrative(source)).toBe(
      'The triage assessment recommends FAST_TRACK with a confidence of 41%. Source triage:0.40971294756240506.',
    );
    expect(narrative('confidence of 1.000.')).toBe('confidence of 100%.');
    expect(narrative('confidence of 1.01; version 0.40971294756240506')).toBe(
      'confidence of 1.01; version 0.40971294756240506',
    );
    expect(source).toContain('confidence of 0.40971294756240506');
  });

  it('removes inline technical citations only when their evidence links are retained', () => {
    const id = 'investigation:16062e46-6cb4-4bba-be3d-8f7d76231cf3';
    const source =
      'The investigation assessment highlights additional damage (dent on the hood) not initially declared, necessitating a human review to assess its impact on the claim and repair cost. The evidence_id is ' +
      id;
    expect(rationaleText(source, [id])).toBe(
      'L’évaluation d’investigation relève des dommages supplémentaires (enfoncement du capot) non déclarés initialement, nécessitant un examen humain de leur incidence sur le dossier et le coût des réparations.',
    );
    expect(rationaleText(source, [])).toContain(id);
    setLocale('en', false);
    expect(rationaleText(source, [id])).toBe(source.split(' The evidence_id is ')[0]);
    const parenthetical = 'Observed damage (evidence_id: investigation:one, vision:two).';
    expect(rationaleText(parenthetical, ['investigation:one', 'vision:two'])).toBe(
      'Observed damage.',
    );
    expect(rationaleText(parenthetical, ['investigation:one'])).toBe(parenthetical);
    expect(source).toContain('The evidence_id is ' + id);
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
