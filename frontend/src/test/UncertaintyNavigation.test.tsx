import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ReviewConsole } from '../App';
import { setLocale } from '../i18n';
import { fullReview, partialReview } from './fixtures';

const hoodSource =
  "The nature and extent of the additional damage on the hood require further investigation and visual confirmation. The 'observations' data within the investigation assessment is truncated, preventing a full understanding of the damage. The exact details of the 'associations' within the 'observations' are missing.";
const recorded = ['Maximum tool steps reached.', hoodSource];
const hoodPoint =
  'A human reviewer must examine the damage to the hood, which is not listed among the declared damaged parts, and assess its impact on the claim and repair cost.';
const stepLimit =
  'The AI agent reached its step limit and produced a recommendation using the information already collected.';
const dataLimit =
  'The saved recommendation reports that some investigation data consulted by the AI was truncated, limiting the analysis.';
const scrollIntoView = vi.fn();
const originalScroll = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'scrollIntoView');

function reviewWithUncertainties(uncertainties = recorded) {
  const view = fullReview();
  view.agent.data!.recommendation!.uncertainties = [...uncertainties];
  return view;
}

beforeEach(() => {
  scrollIntoView.mockClear();
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: scrollIntoView,
  });
});

afterEach(() => {
  if (originalScroll)
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', originalScroll);
  else Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView');
  vi.unstubAllGlobals();
});

describe('Claim verification points and analysis limits', () => {
  it('shows the saved claim point while keeping technical details optional without changing the source', async () => {
    const view = reviewWithUncertainties();
    const original = JSON.stringify(view);
    const fetcher = vi.fn();
    vi.stubGlobal('fetch', fetcher);
    render(<ReviewConsole view={view} />);

    expect(screen.getByText('1 point to verify before making a decision.')).toBeVisible();
    expect(screen.queryByText('2 analysis limits recorded.')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'View analysis limits' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'View points to verify' }));

    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('button', { name: 'Recommendation & signals' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    const points = screen.getByRole('region', { name: 'Points to verify' });
    const list = within(points).getByRole('list', { name: 'Points to verify' });
    expect(points).toHaveFocus();
    expect(list.tagName).toBe('OL');
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((item) => item.textContent),
    ).toEqual([hoodPoint]);
    expect(within(points).queryByText(stepLimit)).not.toBeInTheDocument();
    const details = screen.getByText('Analysis details').closest('details')!;
    expect(details).not.toHaveAttribute('open');
    expect(screen.getByText('Analysis details')).toBeVisible();
    expect(within(details).getByText(stepLimit)).not.toBeVisible();
    expect(within(details).getByText(dataLimit)).not.toBeVisible();
    expect(within(details).getByText('Tools used')).not.toBeVisible();
    expect(within(details).getByText('Recorded uncertainty text')).not.toBeVisible();
    expect(screen.getByText('Human review required:').closest('p')).toBeVisible();
    expect(scrollIntoView).toHaveBeenLastCalledWith(expect.objectContaining({ block: 'start' }));
    expect(scrollIntoView.mock.contexts.at(-1)).toBe(points);
    expect(fetcher).not.toHaveBeenCalled();
    expect(JSON.stringify(view)).toBe(original);
  });

  it('keeps technical-only cases uncluttered while retaining access through analysis details', async () => {
    render(<ReviewConsole view={reviewWithUncertainties(['Maximum tool steps reached.'])} />);
    expect(screen.queryByRole('button', { name: 'View points to verify' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'View analysis limits' })).not.toBeInTheDocument();
    expect(screen.queryByText('1 analysis limit recorded.')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'Evidence' }));
    await userEvent.click(screen.getByRole('button', { name: 'Recommendation & signals' }));
    const summary = screen.getByText('Analysis details');
    expect(summary.tagName).toBe('SUMMARY');
    expect(summary).toBeVisible();
    expect(summary.closest('details')).not.toHaveAttribute('open');
    expect(screen.getByText(stepLimit)).not.toBeVisible();
    await userEvent.click(summary);
    const limits = screen.getByRole('region', { name: 'Analysis limits' });
    expect(summary.closest('details')).toHaveAttribute('open');
    expect(within(limits).getAllByRole('listitem')).toHaveLength(1);
    expect(within(limits).getByText(stepLimit)).toBeVisible();
    expect(screen.queryByRole('list', { name: 'Points to verify' })).not.toBeInTheDocument();
  });

  it('returns to the points after another evidence area has been explored', async () => {
    render(<ReviewConsole view={reviewWithUncertainties()} />);
    await userEvent.click(screen.getByRole('button', { name: 'View points to verify' }));
    await userEvent.click(screen.getByRole('button', { name: 'Photos' }));
    expect(screen.queryByRole('region', { name: 'Points to verify' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'Analysis results' }));
    await userEvent.click(screen.getByRole('button', { name: 'View points to verify' }));
    const points = screen.getByRole('region', { name: 'Points to verify' });
    expect(points).toHaveFocus();
    expect(within(points).getAllByRole('listitem')).toHaveLength(1);
    expect(screen.queryByRole('region', { name: 'Visual Evidence' })).not.toBeInTheDocument();
  });

  it('reveals technical limits and tools only after the reader expands analysis details', async () => {
    render(<ReviewConsole view={reviewWithUncertainties()} />);
    await userEvent.click(screen.getByRole('tab', { name: 'Evidence' }));
    await userEvent.click(screen.getByRole('button', { name: 'Recommendation & signals' }));
    const points = screen.getByRole('region', { name: 'Points to verify' });
    expect(within(points).getAllByRole('listitem')).toHaveLength(1);
    expect(within(points).getByText(hoodPoint)).toBeVisible();
    const summary = screen.getByText('Analysis details');
    const details = summary.closest('details')!;
    expect(details).not.toHaveAttribute('open');
    expect(within(details).getByText(stepLimit)).not.toBeVisible();
    expect(within(details).getByText('Tools used')).not.toBeVisible();
    await userEvent.click(summary);
    const limits = within(details).getByRole('region', { name: 'Analysis limits' });
    expect(within(limits).getAllByRole('listitem')).toHaveLength(2);
    expect(within(limits).getByText(stepLimit)).toBeVisible();
    expect(within(limits).getByText(dataLimit)).toBeVisible();
    expect(within(details).getByText('Tools used')).toBeVisible();
    expect(within(points).getByText(hoodPoint)).toBeVisible();
    await userEvent.click(summary);
    expect(details).not.toHaveAttribute('open');
    expect(within(details).getByText(stepLimit)).not.toBeVisible();
    expect(within(points).getByText(hoodPoint)).toBeVisible();
  });

  it.each([
    ['en', 'View points to verify', 'Analysis details', 'Recorded uncertainty text'],
    ['fr', 'Voir les points à vérifier', 'Détails de l’analyse', 'Textes enregistrés'],
  ] as const)(
    'retains every source entry in an accessible disclosure in %s',
    async (locale, button, analysisSummary, sourceSummary) => {
      setLocale(locale, false);
      render(<ReviewConsole view={reviewWithUncertainties()} />);
      await userEvent.click(screen.getByRole('button', { name: button }));
      const analysisDetails = screen.getByText(analysisSummary).closest('details')!;
      expect(analysisDetails).not.toHaveAttribute('open');
      expect(screen.getByText(analysisSummary)).toBeVisible();
      expect(within(analysisDetails).getByText(sourceSummary)).not.toBeVisible();
      await userEvent.click(within(analysisDetails).getByText(analysisSummary));
      const disclosure = within(analysisDetails).getByText(sourceSummary).closest('details')!;
      expect(disclosure).not.toHaveAttribute('open');
      await userEvent.click(within(disclosure).getByText(sourceSummary));
      expect(disclosure).toHaveAttribute('open');
      for (const source of recorded) expect(within(disclosure).getByText(source)).toBeVisible();
    },
  );

  it('preserves an unknown mixed narrative and visual non-detection as claim points in French', async () => {
    const unknown = 'The tool report is truncated; verify the repair cost with the garage.';
    const nonDetection =
      'Declared damage was not visually confirmed (FRONT_BUMPER); non-detection is not evidence of absence.';
    setLocale('fr', false);
    render(<ReviewConsole view={reviewWithUncertainties([unknown, nonDetection])} />);
    await userEvent.click(screen.getByRole('button', { name: 'Voir les points à vérifier' }));
    const points = screen.getByRole('region', { name: 'Points à vérifier' });
    expect(points).toHaveFocus();
    const list = within(points).getByRole('list', { name: 'Points à vérifier' });
    expect(within(list).getAllByRole('listitem')).toHaveLength(2);
    expect(within(list).getByText(unknown)).toBeVisible();
    expect(within(list).getByText(nonDetection)).toBeVisible();
    expect(screen.queryByRole('region', { name: 'Limites de l’analyse' })).not.toBeInTheDocument();
  });

  it('offers neither shortcut when no completed recommendation is recorded', () => {
    render(<ReviewConsole view={partialReview()} />);
    expect(screen.queryByRole('button', { name: 'View points to verify' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'View analysis limits' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Points to verify' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Analysis limits' })).not.toBeInTheDocument();
  });
});
