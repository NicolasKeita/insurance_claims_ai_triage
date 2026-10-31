export type ReviewStatus =
  'AVAILABLE' | 'UNAVAILABLE' | 'NOT_EXECUTED' | 'FAILED' | 'NO_EVIDENCE' | 'INSUFFICIENT_EVIDENCE';

export interface ReviewSection<T> {
  status: ReviewStatus;
  reason: string | null;
  data: T | null;
  source?: string | null;
  updated_at?: string | null;
}

export type Facts = Record<string, unknown>;

export interface Claim {
  claim_id: string;
  claim_type: string;
  status: string;
  incident: { date: string; location: string; collision_type: string; injuries_declared: boolean };
  vehicle: { make: string; model: string; year: number };
  customer: { customer_id: string };
  policy: { policy_id: string; product: string };
  repair_estimate: { amount: string | number; currency: string };
  declared_damage: string[];
  documents: { type: string; filename: string | null; available: boolean }[];
  images: string[];
}

export interface DocumentItem {
  type: string;
  filename: string | null;
  available: boolean;
  extraction_method?: string | null;
  source?: string | null;
  extracted_data?: Facts | null;
}

export interface Documents {
  items: DocumentItem[];
  extractions?: Record<string, Facts>;
}

export interface Consistency {
  claim_id_match: boolean;
  incident_date_match: boolean;
  location_match: boolean;
  vehicle_match: boolean;
  collision_type_match: boolean;
  injuries_match: boolean;
  declared_damage_match: boolean;
  additional_quote_damage: string[];
  unmapped_quote_items: string[];
}

export interface ModelMetadata {
  registered_model: string;
  model_version: string;
  model_alias?: string;
}

export interface Triage extends ModelMetadata {
  recommended_workflow: string;
  confidence: number;
  probabilities?: Record<string, number>;
}

export interface Anomaly extends ModelMetadata {
  assessment: {
    is_anomalous: boolean;
    anomaly_score: number;
    threshold: number;
    reference_percentile?: number | null;
    unusual_features?: string[];
    statistical_signals?: string[];
  };
}

export interface Signal {
  code: string;
  category: string;
  severity: string;
  title: string;
  explanation: string;
  evidence: Facts;
}

export interface Investigation {
  review_score: number;
  review_priority: string;
  review_recommended: boolean;
  policy_version: string;
  signals: Signal[];
  partial?: boolean;
  unavailable_inputs?: string[];
}

export interface History {
  previous_claim_count: number;
  claims_last_30_days: number;
  claims_last_90_days: number;
  claims_last_365_days: number;
  days_since_previous_claim: number | null;
  previous_anomalous_claim_count: number;
  repair_amount_currency: string | null;
  previous_claims_in_amount_currency: number;
  total_previous_repair_amount: number | string;
  average_previous_repair_amount: number | string | null;
  max_previous_repair_amount: number | string | null;
}

export interface Box {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface Polygon {
  points: { x: number; y: number }[];
  holes?: { x: number; y: number }[][];
  components?: Polygon[];
}

export interface DamageDetection {
  damage_type: string;
  confidence: number;
  bounding_box: Box;
  polygon?: Polygon | null;
}

export interface VisualObservation {
  filename: string;
  damage_detection_index: number;
  visual_damage_type: string;
  damage_confidence: number;
  vehicle_part: string | null;
  part_confidence: number | null;
  containment_ratio: number | null;
  association_status: string;
  damage_detection: DamageDetection;
  mapped_damage_type?: string | null;
}

export interface VisionImage {
  filename: string;
  image_url?: string | null;
  image_unavailable_reason?: string | null;
  damage_assessment?: {
    filename: string;
    width: number;
    height: number;
    detections: DamageDetection[];
  };
  part_assessment?: { detections: { part_type: string; confidence: number; bounding_box: Box }[] };
  evidence: VisualObservation[];
}

export interface Vision {
  images: VisionImage[];
  damage_model_metadata?: Facts;
  part_model_metadata?: Facts;
  fusion_policy_version?: string;
}

export interface VisualConsistency {
  visually_supported_declared_damage: string[];
  visually_supported_quote_damage: string[];
  additional_visual_damage: string[];
  declared_damage_not_visually_confirmed: string[];
  visual_damage_unmapped: VisualObservation[];
}

export interface SimilarClaims {
  results: {
    claim_id: string;
    incident_date: string;
    claim_type: string;
    repair_amount: number | string;
    repair_currency: string;
    similarity_score: number;
  }[];
}

export interface KnowledgeSource {
  chunk_id: string;
  title: string;
  source_type: string;
  section: string;
  text?: string;
  similarity_score?: number | null;
  source_version?: string;
}

export interface Knowledge {
  answer?: string | null;
  insufficient_evidence?: boolean;
  sources: KnowledgeSource[];
}

export interface Agent {
  run_id: string;
  status: string;
  agent_version: string;
  prompt_version: string;
  tools_used: string[];
  created_at?: string;
  completed_at?: string;
  error?: string | null;
  recommendation: {
    recommended_next_action: string;
    case_summary: string;
    rationale: { statement: string; evidence_ids: string[] }[];
    evidence_ids: string[];
    uncertainties: string[];
    human_review_required: boolean;
  } | null;
}

export type SectionKey =
  | 'documents'
  | 'consistency'
  | 'triage'
  | 'anomaly'
  | 'history'
  | 'investigation'
  | 'vision'
  | 'visual_consistency'
  | 'similar_claims'
  | 'knowledge'
  | 'agent';

export interface Evidence {
  id: string;
  section: string;
  title: string;
  data: Facts | null;
  source: string | null;
}

export interface ReviewView {
  claim: Claim;
  mode: 'persisted' | 'demo';
  generated_at: string;
  documents: ReviewSection<Documents>;
  consistency: ReviewSection<Consistency>;
  triage: ReviewSection<Triage>;
  anomaly: ReviewSection<Anomaly>;
  history: ReviewSection<History>;
  investigation: ReviewSection<Investigation>;
  vision: ReviewSection<Vision>;
  visual_consistency: ReviewSection<VisualConsistency>;
  similar_claims: ReviewSection<SimilarClaims>;
  knowledge: ReviewSection<Knowledge>;
  agent: ReviewSection<Agent>;
  evidence: Evidence[];
  audit: {
    timestamps: Facts;
    model_versions: Facts;
    policy_versions: Facts;
    evidence_ids: string[];
    source_documents: string[];
    execution_statuses: Record<string, string>;
  };
}

export interface DemoClaim {
  claim_id: string;
  title: string;
  description: string;
}
