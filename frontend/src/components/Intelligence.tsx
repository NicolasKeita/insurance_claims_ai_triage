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
import { useI18n } from '../i18n';

export function Investigation({
  view,
  onEvidence,
}: {
  view: ReviewView;
  onEvidence: (id: string) => void;
}) {
  const { locale, t, translate } = useI18n();
  return (
    <Section
      id="investigation"
      title={t('Investigation Signals')}
      eyebrow={t('02 / Review context')}
      section={view.investigation}
      className="investigation-panel"
    >
      {(assessment) => (
        <>
          <div className="investigation-summary">
            <Badge tone={assessment.review_priority}>
              {t('{priority} priority', {
                priority:
                  locale === 'fr'
                    ? humanize(assessment.review_priority)
                    : assessment.review_priority,
              })}
            </Badge>
            <span>
              {assessment.signals.length === 1
                ? t('{count} recorded signal', { count: assessment.signals.length })
                : t('{count} recorded signals', { count: assessment.signals.length })}
            </span>
          </div>
          {!assessment.signals.length && (
            <p className="muted">{t('No investigation signals in this recorded assessment.')}</p>
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
                    <h3>{signal.title ? translate(signal.title) : humanize(signal.code)}</h3>
                    <Badge tone={signal.severity}>
                      {locale === 'fr' ? humanize(signal.severity) : signal.severity}
                    </Badge>
                  </div>
                  <p>{translate(signal.explanation)}</p>
                  <p className="signal-meta">
                    <code>{signal.code}</code>
                    <span>{humanize(signal.category)}</span>
                  </p>
                  <details>
                    <summary>{t('Supporting evidence')}</summary>
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
  const { t } = useI18n();
  return (
    <Section
      id="triage"
      title={t('ML Triage')}
      eyebrow={t('03 / Operational routing')}
      section={view.triage}
      help={t(
        'Triage predicts an operational workflow for the claim. It is separate from investigation priority.',
      )}
    >
      {(triage) => (
        <>
          <p className="large-value">{humanize(triage.recommended_workflow)}</p>
          <p className="confidence-label">
            {t('{value} confidence', { value: percent(triage.confidence) })}
          </p>
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
                    aria-label={t('{workflow} class probability', { workflow: humanize(label) })}
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
  const { t, translate } = useI18n();
  return (
    <Section
      id="anomaly"
      title={t('Anomaly Assessment')}
      section={view.anomaly}
      help={t('Anomaly detection identifies unusual patterns; it is not a fraud classifier.')}
    >
      {(anomaly) => (
        <>
          <div className="anomaly-outcome">
            <span>{t('Unusual pattern identified')}</span>
            <Badge tone={anomaly.assessment.is_anomalous ? 'elevated' : 'available'}>
              {anomaly.assessment.is_anomalous ? t('Yes') : t('No')}
            </Badge>
          </div>
          <dl className="metrics three">
            <Metric label={t('Anomaly score')} value={decimal(anomaly.assessment.anomaly_score)} />
            <Metric label={t('Threshold')} value={decimal(anomaly.assessment.threshold)} />
            <Metric
              label={t('Reference percentile')}
              value={percent(anomaly.assessment.reference_percentile)}
            />
          </dl>
          <div className="fact-block">
            <h3>{t('Unusual features')}</h3>
            <Tags
              items={(
                anomaly.assessment.unusual_features ?? anomaly.assessment.statistical_signals
              )?.map(translate)}
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
  const { t } = useI18n();
  return (
    <Section
      id="history"
      title={t('Historical Context')}
      eyebrow={t('04 / Earlier incidents only')}
      section={view.history}
      help={t(
        'Only claims with an earlier incident date are included. Same-day and future incidents are excluded.',
      )}
    >
      {(history) => (
        <>
          <dl className="metrics history-metrics">
            <Metric label={t('Previous claims')} value={history.previous_claim_count} />
            <Metric label={t('Last 30 days')} value={history.claims_last_30_days} />
            <Metric label={t('Last 90 days')} value={history.claims_last_90_days} />
            <Metric label={t('Last 365 days')} value={history.claims_last_365_days} />
            <Metric
              label={t('Days since previous')}
              value={history.days_since_previous_claim ?? t('No previous claim')}
            />
            <Metric
              label={t('Previously anomalous')}
              value={history.previous_anomalous_claim_count}
            />
          </dl>
          <dl className="metrics three repair-history">
            <Metric
              label={t('Previous repair total')}
              value={money(history.total_previous_repair_amount, history.repair_amount_currency)}
            />
            <Metric
              label={t('Average previous repair')}
              value={money(history.average_previous_repair_amount, history.repair_amount_currency)}
            />
            <Metric
              label={t('Highest previous repair')}
              value={money(history.max_previous_repair_amount, history.repair_amount_currency)}
            />
          </dl>
          <p className="section-help">
            {t(
              'Repair amounts use the recorded compatible currency. Historical anomaly flags are stored model outputs.',
            )}
          </p>
        </>
      )}
    </Section>
  );
}
