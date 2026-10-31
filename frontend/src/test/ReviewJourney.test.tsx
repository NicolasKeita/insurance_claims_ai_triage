import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ReviewConsole } from '../App';
import { fullReview, partialReview } from './fixtures';

describe('Progressive claim review', () => {
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
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getAllByText('HUMAN REVIEW REQUIRED').length).toBeGreaterThan(0);
    expect(container).not.toHaveTextContent(hash);
    expect(container).not.toHaveTextContent(sourceId);
    expect(screen.queryByRole('region', { name: 'Documents' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Visual Evidence' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'ML Triage' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Anomaly Assessment' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Audit / Provenance' })).not.toBeInTheDocument();
  });

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
    await userEvent.click(screen.getByRole('tab', { name: 'Overview' }));
    expect(screen.queryByRole('region', { name: 'Visual Evidence' })).not.toBeInTheDocument();
  });

  it('supports keyboard navigation between the optional detail tabs', async () => {
    render(<ReviewConsole view={fullReview()} />);
    screen.getByRole('tab', { name: 'Overview' }).focus();
    await userEvent.keyboard('{ArrowRight}');
    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveFocus();
    expect(screen.getByRole('tab', { name: 'Evidence' })).toHaveAttribute('aria-selected', 'true');
    await userEvent.keyboard('{End}');
    expect(screen.getByRole('tab', { name: 'Technical details' })).toHaveFocus();
    expect(screen.getByRole('region', { name: 'ML Triage' })).toBeVisible();
    await userEvent.keyboard('{Home}');
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveFocus();
    expect(screen.queryByRole('region', { name: 'ML Triage' })).not.toBeInTheDocument();
  });

  it('offers a return to case selection when the parent supplies navigation', async () => {
    const onBack = vi.fn();
    render(
      <ReviewConsole view={partialReview()} scenarioTitle="Coherent documents" onBack={onBack} />,
    );
    expect(screen.getByText('Coherent documents')).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: /back to (demo )?cases/i }));
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
