import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ReviewConsole } from '../App';
import { CaseOverview } from '../components/CaseOverview';
import { setLocale } from '../i18n';
import type { ReviewView } from '../types';
import visualReview from '../../public/demo/reviews/E2E-VISUAL-0001_ANALYSIS-a0f4747edf8c.json';
import { fullReview, partialReview } from './fixtures';

describe('Progressive claim review', () => {
  it.each([
    ['en', 'The investigation assessment highlights additional damage (dent on the hood)'],
    [
      'fr',
      'L’évaluation d’investigation relève des dommages supplémentaires (enfoncement du capot)',
    ],
  ] as const)('foregrounds the saved investigation rationale in %s', (locale, lead) => {
    setLocale(locale, false);
    const view = visualReview as ReviewView;
    const original = JSON.stringify(view);
    const { container } = render(<CaseOverview view={view} onOpenDetails={vi.fn()} />);
    const rationale = screen.getByText((text) => text.startsWith(lead));
    expect(rationale).toBeVisible();
    expect(rationale).toHaveClass('overview-result-text');
    expect(rationale).toHaveTextContent(locale === 'fr' ? 'examen humain' : 'human review');
    const context = container.querySelector('.overview-result-context')!;
    expect(context).toHaveTextContent(
      locale === 'fr'
        ? 'qui ne figure pas parmi les pièces déclarées endommagées'
        : 'which is not listed among the declared damaged parts',
    );
    expect(
      rationale.compareDocumentPosition(context) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(container).not.toHaveTextContent('evidence_id');
    expect(container).not.toHaveTextContent('investigation:16062e46');
    expect(JSON.stringify(view)).toBe(original);
  });

  it.each([
    { origin: 'ai_generated', caption: 'Demo image' },
    { origin: null, caption: 'Demo image' },
  ])(
    'preserves the source image with origin $origin and reports an image failure',
    ({ origin, caption }) => {
      const view = partialReview();
      view.claim.images = ['ai_generated_source.png'];
      const image = {
        filename: 'ai_generated_source.png',
        image_url: '/v1/claims/DEMO-COHERENT/review/images/ai_generated_source.png',
        evidence: [],
      };
      if (origin) Object.assign(image, { provenance: { origin } });
      view.vision = {
        status: 'NOT_EXECUTED',
        reason: 'Vision was not executed.',
        data: { images: [image] },
      };
      const originalVision = JSON.stringify(view.vision);
      render(<CaseOverview view={view} onOpenDetails={vi.fn()} />);
      const preview = screen.getByRole('img', { name: 'Source claim photograph preview' });
      expect(preview).toHaveAttribute('src', image.image_url);
      expect(screen.getByText(caption)).toBeVisible();
      expect(screen.getByText('Vision not executed.')).toBeVisible();
      expect(screen.queryByText('Recorded vision results available.')).not.toBeInTheDocument();
      fireEvent.error(preview);
      expect(screen.getByRole('heading', { name: 'Photos unavailable' })).toBeVisible();
      expect(
        screen.getByText('The registered source photographs could not be loaded.'),
      ).toBeVisible();
      expect(
        screen.queryByRole('img', { name: 'Source claim photograph preview' }),
      ).not.toBeInTheDocument();
      expect(JSON.stringify(view.vision)).toBe(originalVision);
    },
  );

  it.each([
    {
      images: [],
      listed: [],
      heading: 'No photographs listed',
      reason: 'This claim does not list any source photographs.',
    },
    {
      images: [
        {
          filename: 'missing.png',
          image_url: null,
          image_unavailable_reason: 'Original source was not retained.',
          evidence: [],
        },
      ],
      listed: ['missing.png'],
      heading: 'Photos unavailable',
      reason: 'Original source was not retained.',
    },
  ])(
    'shows $heading without substituting an invented photograph',
    ({ images, listed, heading, reason }) => {
      const view = partialReview();
      view.claim.images = listed;
      view.vision = { status: 'UNAVAILABLE', reason: 'Vision unavailable.', data: { images } };
      render(<CaseOverview view={view} onOpenDetails={vi.fn()} />);
      expect(screen.getByRole('heading', { name: heading })).toBeVisible();
      expect(screen.getByText(reason)).toBeVisible();
      expect(screen.queryByRole('img')).not.toBeInTheDocument();
    },
  );

  it('starts with an overview without mounting source or model detail panels', () => {
    const view = fullReview();
    const hash = 'sha256:3aecc5e7c06968a7310d48436739032ee471c0024bdb011a8e3b4d6838843707';
    const sourceId = '08a75c69-6c3f-4aa2-9f98-ebc60bdb754a';
    view.audit.model_versions.triage = hash;
    view.triage.data!.model_version = hash;
    view.agent.data!.run_id = sourceId;
    view.audit.evidence_ids = [`knowledge:${sourceId}`];
    view.knowledge.data!.sources[0].chunk_id = sourceId;
    const { container } = render(<ReviewConsole view={view} />);
    expect(screen.getByRole('tab', { name: 'Analysis results' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    expect(screen.getAllByText('HUMAN REVIEW REQUIRED').length).toBeGreaterThan(0);
    expect(container).not.toHaveTextContent(hash);
    expect(container).not.toHaveTextContent(sourceId);
    expect(screen.queryByRole('region', { name: 'Documents' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Visual Evidence' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'ML Triage' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Anomaly Assessment' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Audit / Provenance' })).not.toBeInTheDocument();
  });

  it('shows the saved conclusion immediately and opens its focused recommendation from the main action', async () => {
    const view = fullReview();
    const original = JSON.stringify(view);
    render(<ReviewConsole view={view} scenarioTitle="Front collision" />);
    expect(screen.getByRole('heading', { name: 'Analysis results' })).toBeVisible();
    expect(screen.getByText(view.investigation.data!.signals[0].explanation)).toBeVisible();
    expect(
      screen.queryByText(view.agent.data!.recommendation!.case_summary),
    ).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Request clarification' })).toBeVisible();
    expect(screen.queryByRole('region', { name: 'Agent Recommendation' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Understand the recommendation' }));
    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('button', { name: 'Recommendation & signals' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    const recommendation = screen.getByRole('region', { name: 'Agent Recommendation' });
    expect(recommendation).toBeVisible();
    expect(recommendation).toHaveFocus();
    expect(recommendation).toHaveTextContent(view.agent.data!.recommendation!.case_summary);
    expect(screen.queryByRole('region', { name: 'Visual Evidence' })).not.toBeInTheDocument();
    expect(JSON.stringify(view)).toBe(original);
  });

  it('uses a completed recommendation summary when there is no investigation signal', () => {
    const view = fullReview();
    view.investigation.data!.signals = [];
    const original = JSON.stringify(view);
    render(<ReviewConsole view={view} />);
    expect(screen.getByRole('heading', { name: 'Analysis results' })).toBeVisible();
    expect(screen.getByText(view.agent.data!.recommendation!.case_summary)).toBeVisible();
    expect(screen.getByRole('button', { name: 'Understand the recommendation' })).toBeVisible();
    expect(JSON.stringify(view)).toBe(original);
  });

  it('opens and focuses the photo results from the overview photo action', async () => {
    render(<ReviewConsole view={fullReview()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Check the photo evidence' }));
    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('button', { name: 'Photos' })).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('region', { name: 'Visual Evidence' })).toHaveFocus();
    expect(screen.getByRole('region', { name: 'Visual Consistency' })).toBeVisible();
    expect(screen.queryByRole('region', { name: 'Agent Recommendation' })).not.toBeInTheDocument();
  });

  it('introduces the workflow before the case title', () => {
    render(<ReviewConsole view={fullReview()} scenarioTitle="Front collision" />);
    const workflow = screen.getByRole('figure', { name: 'Illustrated review workflow' });
    const title = screen.getByRole('heading', { level: 1, name: 'Front collision' });
    expect(workflow.compareDocumentPosition(title) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it.each([
    { sectionStatus: 'FAILED' as const, runStatus: 'COMPLETED' },
    { sectionStatus: 'AVAILABLE' as const, runStatus: 'RUNNING' },
  ])(
    'does not present a conclusion from a $sectionStatus / $runStatus agent as a completed result',
    ({ sectionStatus, runStatus }) => {
      const view = fullReview();
      view.agent.status = sectionStatus;
      view.agent.data!.status = runStatus;
      const original = JSON.stringify(view);
      render(<ReviewConsole view={view} />);
      expect(screen.getByRole('heading', { name: 'No saved recommendation' })).toBeVisible();
      expect(
        screen.queryByText(view.agent.data!.recommendation!.case_summary),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole('heading', { name: 'Request clarification' }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole('button', { name: 'Understand the recommendation' }),
      ).not.toBeInTheDocument();
      expect(JSON.stringify(view)).toBe(original);
    },
  );

  it('opens one evidence group at a time and leaves other details unmounted', async () => {
    render(<ReviewConsole view={fullReview()} />);
    await userEvent.click(screen.getByRole('tab', { name: 'Evidence' }));
    await userEvent.click(screen.getByRole('button', { name: 'Documents' }));
    expect(screen.getByRole('region', { name: 'Documents' })).toBeVisible();
    expect(screen.queryByRole('region', { name: 'Visual Evidence' })).not.toBeInTheDocument();
    expect(
      screen.queryByRole('region', { name: 'Policy & Procedure Evidence' }),
    ).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Photos' }));
    expect(screen.getByRole('region', { name: 'Visual Evidence' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Documents' })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    expect(screen.queryByRole('region', { name: 'Documents' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'Analysis results' }));
    expect(screen.queryByRole('region', { name: 'Visual Evidence' })).not.toBeInTheDocument();
  });

  it('supports keyboard navigation between the optional detail tabs', async () => {
    render(<ReviewConsole view={fullReview()} />);
    screen.getByRole('tab', { name: 'Analysis results' }).focus();
    await userEvent.keyboard('{ArrowRight}');
    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveFocus();
    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveAttribute('aria-selected', 'true');
    await userEvent.keyboard('{End}');
    expect(screen.getByRole('tab', { name: 'Technical details' })).toHaveFocus();
    expect(screen.getByRole('region', { name: 'ML Triage' })).toBeVisible();
    await userEvent.keyboard('{Home}');
    expect(screen.getByRole('tab', { name: 'Analysis results' })).toHaveFocus();
    expect(screen.queryByRole('region', { name: 'ML Triage' })).not.toBeInTheDocument();
  });

  it('offers optional case exploration when the parent supplies navigation', async () => {
    const onBack = vi.fn();
    render(
      <ReviewConsole view={partialReview()} scenarioTitle="Coherent documents" onBack={onBack} />,
    );
    expect(screen.getByText('Coherent documents')).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Explore other cases' }));
    expect(onBack).toHaveBeenCalledOnce();
  });

  it('opens the source details requested from the case overview', async () => {
    render(<ReviewConsole view={fullReview()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Review documents' }));
    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('button', { name: 'Documents' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    expect(screen.getByRole('region', { name: 'Documents' })).toBeVisible();
    expect(screen.queryByRole('region', { name: 'Agent Recommendation' })).not.toBeInTheDocument();
  });

  it('retains an unavailable recommendation without inventing an action', async () => {
    render(<ReviewConsole view={partialReview()} />);
    expect(screen.getByRole('heading', { name: 'No saved recommendation' })).toBeVisible();
    expect(screen.queryByText('Request clarification')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'Evidence' }));
    await userEvent.click(screen.getByRole('button', { name: 'Recommendation & signals' }));
    const recommendation = screen.getByRole('region', { name: 'Agent Recommendation' });
    expect(recommendation).toHaveTextContent('Not executed');
    expect(recommendation).toHaveTextContent('No stored assessment is available.');
    expect(recommendation).not.toHaveTextContent('Request clarification');
    expect(recommendation).not.toHaveTextContent('Continue standard review');
  });
});
