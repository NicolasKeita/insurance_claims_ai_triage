import type { ReviewSection, ReviewView } from '../types';

export function unavailable<T>(
  status: ReviewSection<T>['status'] = 'NOT_EXECUTED',
): ReviewSection<T> {
  return { status, reason: 'No stored assessment is available.', data: null };
}

export function available<T>(data: T): ReviewSection<T> {
  return {
    status: 'AVAILABLE',
    reason: null,
    data,
    source: 'Controlled test record',
    updated_at: '2026-01-20T12:00:00Z',
  };
}

export function partialReview(): ReviewView {
  return {
    claim: {
      claim_id: 'DEMO-COHERENT',
      claim_type: 'AUTO_COLLISION',
      status: 'OPEN',
      incident: {
        date: '2026-01-20',
        location: 'Lyon',
        collision_type: 'REAR_END',
        injuries_declared: false,
      },
      vehicle: { make: 'Peugeot', model: '208', year: 2021 },
      customer: { customer_id: 'CUSTOMER-001' },
      policy: { policy_id: 'POLICY-001', product: 'AUTO_BASIC' },
      repair_estimate: { amount: '1240', currency: 'EUR' },
      declared_damage: ['FRONT_BUMPER'],
      documents: [{ type: 'CLAIM_FORM', filename: 'claim_form.pdf', available: true }],
      images: [],
    },
    mode: 'demo',
    generated_at: '2026-01-20T12:00:00Z',
    documents: available({
      items: [
        {
          type: 'CLAIM_FORM',
          filename: 'claim_form.pdf',
          available: true,
          extraction_method: 'native',
          source: 'Controlled fixture',
          extracted_data: { vehicle_make: 'Peugeot', incident_date: '2026-01-20' },
        },
      ],
    }),
    consistency: unavailable(),
    triage: unavailable(),
    anomaly: unavailable('UNAVAILABLE'),
    history: unavailable(),
    investigation: unavailable(),
    vision: unavailable(),
    visual_consistency: unavailable(),
    similar_claims: unavailable(),
    knowledge: unavailable('UNAVAILABLE'),
    agent: unavailable(),
    evidence: [],
    audit: {
      timestamps: {},
      model_versions: {},
      policy_versions: {},
      evidence_ids: [],
      source_documents: ['claim_form.pdf'],
      execution_statuses: { anomaly: 'UNAVAILABLE' },
    },
  };
}

export function fullReview(): ReviewView {
  const view = partialReview();
  view.consistency = available({
    claim_id_match: true,
    incident_date_match: true,
    location_match: true,
    vehicle_match: true,
    collision_type_match: true,
    injuries_match: true,
    declared_damage_match: true,
    additional_quote_damage: ['HOOD'],
    unmapped_quote_items: ['special paint treatment'],
  });
  view.triage = available({
    recommended_workflow: 'MANUAL_REVIEW',
    confidence: 0.84,
    probabilities: { MANUAL_REVIEW: 0.84, STANDARD: 0.16 },
    registered_model: 'triage-candidate',
    model_version: '3',
  });
  view.anomaly = available({
    assessment: {
      is_anomalous: true,
      anomaly_score: -0.21,
      threshold: -0.18,
      reference_percentile: 0.97,
      statistical_signals: ['Unusually high repair estimate'],
    },
    registered_model: 'anomaly-candidate',
    model_version: '4',
  });
  view.investigation = available({
    review_score: 35,
    review_priority: 'ELEVATED',
    review_recommended: true,
    policy_version: 'investigation-v2',
    signals: [
      {
        code: 'ADDITIONAL_QUOTE_DAMAGE',
        category: 'DOCUMENT_CONSISTENCY',
        severity: 'MEDIUM',
        title: 'Additional repair quote damage',
        explanation: 'The quote includes a hood replacement beyond the declared bumper damage.',
        evidence: { additional_quote_damage: ['HOOD'] },
      },
    ],
  });
  view.history = available({
    previous_claim_count: 2,
    claims_last_30_days: 0,
    claims_last_90_days: 1,
    claims_last_365_days: 2,
    days_since_previous_claim: 60,
    previous_anomalous_claim_count: 1,
    repair_amount_currency: 'EUR',
    previous_claims_in_amount_currency: 2,
    total_previous_repair_amount: 2000,
    average_previous_repair_amount: 1000,
    max_previous_repair_amount: 1300,
  });
  view.vision = {
    status: 'NO_EVIDENCE',
    reason: 'No supported visual observations were retained.',
    data: {
      images: [
        {
          filename: 'image_01.jpg',
          image_url: null,
          image_unavailable_reason: 'Original photograph was not copied into the demo.',
          damage_assessment: { filename: 'image_01.jpg', width: 800, height: 600, detections: [] },
          evidence: [],
        },
      ],
    },
  };
  view.visual_consistency = available({
    visually_supported_declared_damage: [],
    visually_supported_quote_damage: [],
    additional_visual_damage: [],
    declared_damage_not_visually_confirmed: ['FRONT_BUMPER'],
    visual_damage_unmapped: [],
  });
  view.similar_claims = available({
    results: [
      {
        claim_id: 'PREVIOUS-001',
        incident_date: '2025-11-21',
        claim_type: 'AUTO_COLLISION',
        repair_amount: 1300,
        repair_currency: 'EUR',
        similarity_score: 0.87,
      },
    ],
  });
  view.knowledge = available({
    answer: 'Request clarification of additional quoted repairs.',
    insufficient_evidence: false,
    sources: [
      {
        chunk_id: 'chunk-123',
        title: 'Motor claim handling procedure',
        source_type: 'PROCEDURE',
        section: 'Repair clarification',
        text: 'Clarify differences between declared damage and quoted repairs.',
        similarity_score: 0.88,
      },
    ],
  });
  view.agent = available({
    run_id: 'run-001',
    status: 'COMPLETED',
    agent_version: '1.0',
    prompt_version: 'v3',
    tools_used: ['GET_INVESTIGATION', 'SEARCH_KNOWLEDGE'],
    recommendation: {
      recommended_next_action: 'REQUEST_CLARIFICATION',
      case_summary: 'The submitted quote includes an additional hood repair.',
      rationale: [
        {
          statement: 'The recorded procedure recommends clarification.',
          evidence_ids: ['knowledge:chunk-123'],
        },
      ],
      evidence_ids: ['knowledge:chunk-123'],
      uncertainties: ['The repair need requires human confirmation.'],
      human_review_required: true,
    },
  });
  view.evidence = [
    {
      id: 'knowledge:chunk-123',
      section: 'knowledge',
      title: 'Motor claim handling procedure',
      data: { source_type: 'PROCEDURE', section: 'Repair clarification', chunk_id: 'chunk-123' },
      source: 'auto_claim_handling_v1',
    },
  ];
  view.audit.model_versions = { triage: '3', anomaly: '4', agent: '1.0' };
  view.audit.policy_versions = { investigation: 'investigation-v2', prompt: 'v3' };
  view.audit.evidence_ids = ['knowledge:chunk-123'];
  return view;
}
