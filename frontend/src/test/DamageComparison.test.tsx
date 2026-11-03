import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import visualReview from '../../public/demo/reviews/E2E-VISUAL-0001_ANALYSIS-a0f4747edf8c.json';
import { DamageComparison } from '../components/DamageComparison';
import { setLocale } from '../i18n';
import type { ReviewView } from '../types';
import { partialReview } from './fixtures';

describe('Declared and additional damage context', () => {
  it('shows the declared bumper alongside the additional hood observation in French', () => {
    setLocale('fr', false);
    const view = visualReview as ReviewView;
    const original = JSON.stringify(view);
    render(<DamageComparison view={view} />);
    const comparison = screen.getByLabelText('Dommages déclarés et supplémentaires');
    expect(within(comparison).getByText('Dommages déclarés')).toBeVisible();
    expect(within(comparison).getByText('Pare-chocs avant')).toBeVisible();
    expect(
      within(comparison).getByText('Dommages supplémentaires observés sur les photos (à vérifier)'),
    ).toBeVisible();
    expect(within(comparison).getByText('Capot')).toBeVisible();
    expect(JSON.stringify(view)).toBe(original);
  });

  it.each(['FAILED', 'UNAVAILABLE', 'NOT_EXECUTED', 'NO_EVIDENCE'] as const)(
    'ignores stored visual data when its status is %s',
    (status) => {
      const view = structuredClone(visualReview) as ReviewView;
      view.visual_consistency.status = status;
      render(<DamageComparison view={view} />);
      expect(screen.getByText('Front bumper')).toBeVisible();
      expect(screen.queryByText('Hood')).not.toBeInTheDocument();
      expect(
        screen.queryByText('Additional damage observed in photographs (to verify)'),
      ).not.toBeInTheDocument();
    },
  );

  it('keeps missing declared and visual data explicit without inventing a no-damage finding', () => {
    const view = partialReview();
    view.claim.declared_damage = [];
    view.visual_consistency = { status: 'AVAILABLE', reason: null, data: null };
    render(<DamageComparison view={view} />);
    expect(screen.getByText('Declared damage')).toBeVisible();
    expect(screen.getByText('Not recorded')).toBeVisible();
    expect(
      screen.queryByText('Additional damage observed in photographs (to verify)'),
    ).not.toBeInTheDocument();
  });
});
