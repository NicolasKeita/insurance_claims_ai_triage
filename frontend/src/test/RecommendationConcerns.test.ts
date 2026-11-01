import { describe, expect, it } from 'vitest';
import { recommendationConcerns } from '../review/recommendationConcerns';

const hoodSource =
  "The nature and extent of the additional damage on the hood require further investigation and visual confirmation. The 'observations' data within the investigation assessment is truncated, preventing a full understanding of the damage. The exact details of the 'associations' within the 'observations' are missing.";

describe('Recommendation concerns from saved uncertainty text', () => {
  it('separates the saved hood verification from both recorded technical limitations', () => {
    const source = Object.freeze(['Maximum tool steps reached.', hoodSource]);
    expect(recommendationConcerns(source)).toEqual({
      points: ['Confirm the presence and extent of the additional damage to the hood.'],
      limits: [
        'The AI agent reached its step limit and produced a recommendation using the information already collected.',
        'The saved recommendation reports that some investigation data consulted by the AI was truncated, limiting the analysis.',
      ],
    });
    expect(source).toEqual(['Maximum tool steps reached.', hoodSource]);
  });

  it('classifies the supported context and comparison gaps as analysis limits', () => {
    expect(
      recommendationConcerns([
        'Evidence details were shortened to fit the model context.',
        'FIND_SIMILAR_CLAIMS: No eligible similar claims were retrieved',
      ]),
    ).toEqual({
      points: [],
      limits: [
        "Some evidence details were shortened to fit the AI model's context.",
        'No eligible similar claims were found for comparison.',
      ],
    });
  });

  it('keeps unknown mixed text and non-detections intact rather than guessing a category', () => {
    const source = [
      'The tool report is truncated; verify the repair cost with the garage.',
      'Maximum tool steps reached. The repair estimate requires confirmation.',
      'Declared damage was not visually confirmed (FRONT_BUMPER); non-detection is not evidence of absence.',
    ];
    expect(recommendationConcerns(source)).toEqual({ points: source, limits: [] });
  });

  it('supports a recommendation with no recorded uncertainties', () => {
    expect(recommendationConcerns([])).toEqual({ points: [], limits: [] });
  });
});
