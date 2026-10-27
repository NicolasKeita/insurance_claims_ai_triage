import {
  Badge,
  EvidenceLinks,
  FactList,
  Metric,
  Section,
  Tags,
  decimal,
  humanize,
  money,
  percent,
} from '../presentation';
import type { ReviewView } from '../types';

export function Investigation({
  view,
  onEvidence,
}: {
  view: ReviewView;
  onEvidence: (id: string) => void;
}) {
  return (
    <Section
      id="investigation"
      title="Investigation Signals"
      eyebrow="02 / Review context"
      section={view.investigation}
      className="investigation-panel"
    >
      {(assessment) => (
        <>
          <div className="investigation-summary">
            <Badge tone={assessment.review_priority}>{assessment.review_priority} priority</Badge>
            <span>
              {assessment.signals.length} recorded{' '}
              {assessment.signals.length === 1 ? 'signal' : 'signals'}
            </span>
          </div>
          {!assessment.signals.length && (
            <p className="muted">No investigation signals in this recorded assessment.</p>
          )}
          <div className="signal-list">
            {assessment.signals.map((signal) => {
              const evidence = view.evidence.filter((item) => item.id.includes(signal.code));
              return (
                <article className="signal" key={signal.code}>
                  <div className="signal-top">
                    <span
                      className={`severity-marker ${signal.severity.toLowerCase()}`}
                      aria-hidden="true"
                    />
                    <h3>{signal.title || humanize(signal.code)}</h3>
                    <Badge tone={signal.severity}>{signal.severity}</Badge>
                  </div>
                  <p>{signal.explanation}</p>
                  <p className="signal-meta">
                    <code>{signal.code}</code>
                    <span>{humanize(signal.category)}</span>
                  </p>
                  <details>
                    <summary>Supporting evidence</summary>
                    <FactList facts={signal.evidence} />
                    <EvidenceLinks ids={evidence.map((item) => item.id)} onOpen={onEvidence} />
                  </details>
                </article>
              );
            })}
          </div>
        </>
      )}
    </Section>
  );
}

export function Triage({ view }: { view: ReviewView }) {
  return (
    <Section
      id="triage"
      title="ML Triage"
      eyebrow="03 / Operational routing"
      section={view.triage}
      help="Triage predicts an operational workflow for the claim. It is separate from investigation priority."
    >
      {(triage) => (
        <>
          <p className="large-value">{humanize(triage.recommended_workflow)}</p>
          <p className="confidence-label">{percent(triage.confidence)} confidence</p>
          {triage.probabilities && (
            <div className="probabilities">
              {Object.entries(triage.probabilities).map(([label, value]) => (
                <div key={label}>
                  <div>
                    <span>{humanize(label)}</span>
                    <strong>{percent(value)}</strong>
                  </div>
                  <meter
                    min={0}
                    max={1}
                    value={value}
                    aria-label={`${humanize(label)} class probability`}
                  />
                </div>
              ))}
            </div>
          )}
          <p className="model-note">
            {triage.registered_model} · v{triage.model_version}
          </p>
        </>
      )}
    </Section>
  );
}

export function Anomaly({ view }: { view: ReviewView }) {
  return (
    <Section
      id="anomaly"
      title="Anomaly Assessment"
      section={view.anomaly}
      help="Anomaly detection identifies unusual patterns; it is not a fraud classifier."
    >
      {(anomaly) => (
        <>
          <div className="anomaly-outcome">
            <span>Unusual pattern identified</span>
            <Badge tone={anomaly.assessment.is_anomalous ? 'elevated' : 'available'}>
              {anomaly.assessment.is_anomalous ? 'Yes' : 'No'}
            </Badge>
          </div>
          <dl className="metrics three">
            <Metric label="Anomaly score" value={decimal(anomaly.assessment.anomaly_score)} />
            <Metric label="Threshold" value={decimal(anomaly.assessment.threshold)} />
            <Metric
              label="Reference percentile"
              value={percent(anomaly.assessment.reference_percentile)}
            />
          </dl>
          <div className="fact-block">
            <h3>Unusual features</h3>
            <Tags
              items={anomaly.assessment.unusual_features ?? anomaly.assessment.statistical_signals}
            />
          </div>
          <p className="model-note">
            {anomaly.registered_model} · v{anomaly.model_version}
          </p>
        </>
      )}
    </Section>
  );
}

export function ClaimHistory({ view }: { view: ReviewView }) {
  return (
    <Section
      id="history"
      title="Historical Context"
      eyebrow="04 / Earlier incidents only"
      section={view.history}
      help="Only claims with an earlier incident date are included. Same-day and future incidents are excluded."
    >
      {(history) => (
        <>
          <dl className="metrics history-metrics">
            <Metric label="Previous claims" value={history.previous_claim_count} />
            <Metric label="Last 30 days" value={history.claims_last_30_days} />
            <Metric label="Last 90 days" value={history.claims_last_90_days} />
            <Metric label="Last 365 days" value={history.claims_last_365_days} />
            <Metric
              label="Days since previous"
              value={history.days_since_previous_claim ?? 'No previous claim'}
            />
            <Metric label="Previously anomalous" value={history.previous_anomalous_claim_count} />
          </dl>
          <dl className="metrics three repair-history">
            <Metric
              label="Previous repair total"
              value={money(history.total_previous_repair_amount, history.repair_amount_currency)}
            />
            <Metric
              label="Average previous repair"
              value={money(history.average_previous_repair_amount, history.repair_amount_currency)}
            />
            <Metric
              label="Highest previous repair"
              value={money(history.max_previous_repair_amount, history.repair_amount_currency)}
            />
          </dl>
          <p className="section-help">
            Repair amounts use the recorded compatible currency. Historical anomaly flags are stored
            model outputs.
          </p>
        </>
      )}
    </Section>
  );
}
