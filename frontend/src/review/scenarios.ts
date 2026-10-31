import type { DemoClaim } from '../types';

const descriptions: Record<string, { title: string; description: string; focus: string }> = {
  'E2E-COHERENT-0001': {
    title: 'Matching documents',
    description:
      'Follow a claim whose documents agree, then inspect the recommendation and its limits.',
    focus: 'Document comparison',
  },
  'E2E-QUOTE-0001': {
    title: 'Extra repair in the quote',
    description: 'See how an additional repair item is explained and linked to its source.',
    focus: 'Review signals',
  },
  'E2E-MISSING-0001': {
    title: 'A missing document',
    description: 'Explore an incomplete case and see which conclusions remain unavailable.',
    focus: 'Partial evidence',
  },
  'E2E-VISUAL-0001': {
    title: 'Photo evidence',
    description: 'Inspect recorded damage observations and trace them back to the original photo.',
    focus: 'Computer vision',
  },
};

/** Editorial labels only; all claim facts and assessments still come from the API. */
export function scenarioCopy(claim: DemoClaim) {
  return (
    descriptions[claim.claim_id] ?? {
      title: claim.title,
      description:
        'Open this prepared case and explore the evidence behind its recorded assessment.',
      focus: 'Claim review',
    }
  );
}
