import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ReviewConsole } from '../App';
import { setLocale } from '../i18n';
import { fullReview, partialReview } from './fixtures';

const recorded = ['Maximum tool steps reached.', 'The repair need requires human confirmation.'];
const scrollIntoView = vi.fn();
const originalScroll = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'scrollIntoView');

function reviewWithTwoUncertainties() {
  const view = fullReview();
  view.agent.data!.recommendation!.uncertainties = [...recorded];
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

describe('Direct uncertainty navigation', () => {
  it('reveals the exact two recorded uncertainties and focuses their list without another request', async () => {
    const view = reviewWithTwoUncertainties();
    const original = JSON.stringify(view);
    const fetcher = vi.fn();
    vi.stubGlobal('fetch', fetcher);
    render(<ReviewConsole view={view} />);
    expect(
      screen.getByText('2 recorded uncertainties. Review them before making a decision.'),
    ).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'View uncertainties' }));
    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('button', { name: 'Recommendation & signals' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    const region = screen.getByRole('region', { name: /^Uncertainties to review/ });
    const list = within(region).getByRole('list', { name: 'Recorded uncertainties' });
    expect(region).toHaveFocus();
    expect(list.tagName).toBe('OL');
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((item) => item.textContent),
    ).toEqual(recorded);
    for (const item of within(list).getAllByRole('listitem')) expect(item).toBeVisible();
    expect(scrollIntoView).toHaveBeenLastCalledWith(expect.objectContaining({ block: 'start' }));
    expect(scrollIntoView.mock.contexts.at(-1)).toBe(region);
    expect(fetcher).not.toHaveBeenCalled();
    expect(JSON.stringify(view)).toBe(original);
  });

  it('returns to the uncertainty list after another evidence area has been explored', async () => {
    render(<ReviewConsole view={reviewWithTwoUncertainties()} />);
    await userEvent.click(screen.getByRole('button', { name: 'View uncertainties' }));
    await userEvent.click(screen.getByRole('button', { name: 'Photos' }));
    expect(
      screen.queryByRole('region', { name: /^Uncertainties to review/ }),
    ).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'Analysis results' }));
    await userEvent.click(screen.getByRole('button', { name: 'View uncertainties' }));
    const region = screen.getByRole('region', { name: /^Uncertainties to review/ });
    expect(region).toHaveFocus();
    expect(within(region).getAllByRole('listitem')).toHaveLength(2);
    expect(screen.queryByRole('region', { name: 'Visual Evidence' })).not.toBeInTheDocument();
  });

  it('shows the same numbered items when recommendation details are opened directly', async () => {
    render(<ReviewConsole view={reviewWithTwoUncertainties()} />);
    await userEvent.click(screen.getByRole('tab', { name: 'Evidence' }));
    await userEvent.click(screen.getByRole('button', { name: 'Recommendation & signals' }));
    const region = screen.getByRole('region', { name: /^Uncertainties to review/ });
    const list = within(region).getByRole('list', { name: 'Recorded uncertainties' });
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((item) => item.textContent),
    ).toEqual(recorded);
  });

  it('labels and focuses the French uncertainty section while preserving unknown source text', async () => {
    setLocale('fr', false);
    render(<ReviewConsole view={reviewWithTwoUncertainties()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Voir les incertitudes' }));
    const region = screen.getByRole('region', { name: /^Incertitudes à examiner/ });
    const list = within(region).getByRole('list', { name: 'Incertitudes enregistrées' });
    expect(region).toHaveFocus();
    expect(within(list).getAllByRole('listitem')).toHaveLength(2);
    expect(
      within(list).getByText('Le nombre maximal d’étapes d’utilisation des outils a été atteint.'),
    ).toBeVisible();
    expect(within(list).getByText(recorded[1])).toBeVisible();
  });

  it('offers no uncertainty shortcut when no completed recommendation is recorded', () => {
    render(<ReviewConsole view={partialReview()} />);
    expect(screen.queryByRole('button', { name: 'View uncertainties' })).not.toBeInTheDocument();
    expect(
      screen.queryByRole('region', { name: /^Uncertainties to review/ }),
    ).not.toBeInTheDocument();
  });
});
