type ConcernPresentation = { point?: string; limit: string };

// Older saved recommendations combine claim uncertainties and execution limits.
// Only these exact known messages have a curated presentation. Unknown narratives
// stay intact as verification points; no keyword guessing or source mutation.
const presentations = new Map<string, ConcernPresentation>([
  [
    'Maximum tool steps reached.',
    {
      limit:
        'The AI agent reached its step limit and produced a recommendation using the information already collected.',
    },
  ],
  [
    'Evidence details were shortened to fit the model context.',
    { limit: "Some evidence details were shortened to fit the AI model's context." },
  ],
  [
    'FIND_SIMILAR_CLAIMS: No eligible similar claims were retrieved',
    { limit: 'No eligible similar claims were found for comparison.' },
  ],
  [
    "The nature and extent of the additional damage on the hood require further investigation and visual confirmation. The 'observations' data within the investigation assessment is truncated, preventing a full understanding of the damage. The exact details of the 'associations' within the 'observations' are missing.",
    {
      point:
        'A human reviewer must examine the damage to the hood, which is not listed among the declared damaged parts, and assess its impact on the claim and repair cost.',
      limit:
        'The saved recommendation reports that some investigation data consulted by the AI was truncated, limiting the analysis.',
    },
  ],
]);

export function recommendationConcerns(uncertainties: readonly string[]): {
  points: string[];
  limits: string[];
} {
  const points: string[] = [];
  const limits: string[] = [];
  for (const source of uncertainties) {
    const presentation = presentations.get(source);
    if (!presentation) points.push(source);
    else {
      if (presentation.point) points.push(presentation.point);
      limits.push(presentation.limit);
    }
  }
  return { points, limits };
}
