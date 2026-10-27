import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { ReviewConsole } from '../App';
import { anchor } from '../presentation';
import { available, fullReview, partialReview, unavailable } from './fixtures';
import type { ReviewStatus } from '../types';

describe('Claim Review', () => {
  it('shows assessment scope even when a partial result is available', () => {
    const view = fullReview();
    view.investigation.reason =
      'Partial prioritization from document and completeness evidence only; anomaly, history and vision not executed.';
    render(<ReviewConsole view={view} />);
    expect(screen.getByText(view.investigation.reason)).toBeVisible();
  });
  it('renders claim identity, source documents and extracted fields', async () => {
    render(<ReviewConsole view={partialReview()} />);
    expect(screen.getByRole('heading', { level: 1, name: 'DEMO-COHERENT' })).toBeInTheDocument();
    expect(screen.getByText('Peugeot 208')).toBeInTheDocument();
    expect(screen.getByText('Lyon')).toBeInTheDocument();
    expect(screen.getAllByText('claim_form.pdf').length).toBeGreaterThan(0);
    await userEvent.click(screen.getByText('Extracted fields'));
    expect(screen.getByText('Vehicle make')).toBeVisible();
    expect(screen.getByText('Peugeot')).toBeVisible();
    expect(screen.getByText('Native')).toBeInTheDocument();
  });

  it.each<ReviewStatus>([
    'UNAVAILABLE',
    'NOT_EXECUTED',
    'FAILED',
    'NO_EVIDENCE',
    'INSUFFICIENT_EVIDENCE',
  ])('preserves %s as an explicit state rather than a negative assessment', (status) => {
    const view = partialReview();
    view.anomaly = unavailable(status);
    render(<ReviewConsole view={view} />);
    const section = screen.getByRole('region', { name: 'Anomaly Assessment' });
    expect(within(section).getByRole('status')).toHaveTextContent(
      'No stored assessment is available.',
    );
    expect(within(section).queryByText('No anomaly')).not.toBeInTheDocument();
    expect(within(section).queryByText('0')).not.toBeInTheDocument();
    expect(within(section).queryByText('Anomaly score')).not.toBeInTheDocument();
  });

  it('renders investigation facts, workflow and anomaly as separate assessments', async () => {
    render(<ReviewConsole view={fullReview()} />);
    expect(screen.getByText('ELEVATED priority')).toBeInTheDocument();
    expect(screen.getByText('ADDITIONAL_QUOTE_DAMAGE')).toBeInTheDocument();
    expect(screen.getByText('MEDIUM')).toBeInTheDocument();
    expect(screen.getByText(/quote includes a hood replacement/)).toBeInTheDocument();
    await userEvent.click(screen.getByText('Supporting evidence'));
    expect(screen.getAllByText('HOOD').length).toBeGreaterThan(0);
    expect(screen.getByText('84% confidence')).toBeInTheDocument();
    expect(screen.getByText('Unusually high repair estimate')).toBeInTheDocument();
    expect(
      screen.getByText('Review prioritization — not a fraud probability.'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Anomaly detection identifies unusual patterns; it is not a fraud classifier.',
      ),
    ).toBeInTheDocument();
  });

  it('opens the exact retained agent evidence and supports keyboard dismissal', async () => {
    render(<ReviewConsole view={fullReview()} />);
    const agent = screen.getByRole('region', { name: 'Agent Recommendation' });
    const evidence = within(agent).getAllByRole('button', { name: /knowledge:chunk-123/ });
    await userEvent.click(evidence[0]);
    const dialog = screen.getByRole('dialog', { name: 'Motor claim handling procedure' });
    expect(within(dialog).getByText('auto_claim_handling_v1')).toBeVisible();
    expect(document.getElementById(anchor('knowledge:chunk-123'))).toHaveClass(
      'evidence-highlight',
    );
    expect(within(dialog).getByRole('button', { name: 'Close evidence source' })).toHaveFocus();
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('retains non-detection uncertainty and explains unavailable photographs', () => {
    render(<ReviewConsole view={fullReview()} />);
    expect(
      screen.getByText('Not visually confirmed does not mean the declared damage is absent.'),
    ).toBeVisible();
    expect(screen.getByText('Original photograph was not copied into the demo.')).toBeVisible();
    expect(screen.getByText('No supported visual observations were retained.')).toBeVisible();
    expect(screen.getByText('No stored observations for this image.')).toBeVisible();
  });

  it('labels an unassessed source photograph without implying a recorded overlay', () => {
    const view = partialReview();
    view.vision = {
      status: 'NOT_EXECUTED',
      reason: 'Vision has not been executed for this image.',
      data: {
        images: [
          {
            filename: 'source.jpg',
            image_url: '/v1/claims/DEMO-COHERENT/review/images/source.jpg',
            evidence: [],
          },
        ],
      },
    };
    render(<ReviewConsole view={view} />);
    expect(screen.getByRole('img', { name: 'Claim photograph: source.jpg' })).toBeVisible();
    expect(screen.getByText('Vision has not been executed for this image.')).toBeVisible();
    expect(screen.getByText('Source photograph · no recorded overlay')).toBeVisible();
    expect(screen.queryByText('Recorded evidence overlay')).not.toBeInTheDocument();
    expect(
      screen.queryByRole('img', { name: 'Recorded damage masks and bounding boxes' }),
    ).not.toBeInTheDocument();
  });

  it('shows actual stored image geometry and reports a failed image load', () => {
    const view = fullReview();
    view.vision = available({
      images: [
        {
          filename: 'photo.jpg',
          image_url: '/v1/claims/DEMO-COHERENT/images/photo.jpg',
          damage_assessment: {
            filename: 'photo.jpg',
            width: 100,
            height: 100,
            detections: [
              {
                damage_type: 'DENT',
                confidence: 0.8,
                bounding_box: { x_min: 0.1, y_min: 0.2, x_max: 0.3, y_max: 0.4 },
              },
            ],
          },
          evidence: [],
        },
      ],
    });
    render(<ReviewConsole view={view} />);
    const photo = screen.getByRole('img', { name: 'Claim photograph: photo.jpg' });
    expect(photo).toHaveAttribute('src', '/v1/claims/DEMO-COHERENT/images/photo.jpg');
    expect(
      screen
        .getByRole('img', { name: 'Recorded damage masks and bounding boxes' })
        .querySelector('rect'),
    ).toHaveAttribute('x', '10');
    fireEvent.error(photo);
    expect(screen.getByText('The image file could not be loaded.')).toBeInTheDocument();
  });

  it('distinguishes procedures from policies and uses Similarity for retrieved claims', () => {
    render(<ReviewConsole view={fullReview()} />);
    expect(screen.getByText('PROCEDURE')).toBeInTheDocument();
    expect(screen.getByText('Internal guidance')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Similarity' })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Probability' })).not.toBeInTheDocument();
  });

  it('makes human oversight explicit without automatic decisions or unsupported conclusions', () => {
    const { container } = render(<ReviewConsole view={fullReview()} />);
    expect(screen.getAllByText('HUMAN REVIEW REQUIRED').length).toBe(2);
    expect(screen.getByText('Human review required:')).toBeInTheDocument();
    expect(container).not.toHaveTextContent(
      /fraud detected|fraud score|automatically detects fraud|confirmed fraud/i,
    );
    expect(
      screen.queryByRole('button', { name: /approve|deny|pay|run|generate/i }),
    ).not.toBeInTheDocument();
  });
});
