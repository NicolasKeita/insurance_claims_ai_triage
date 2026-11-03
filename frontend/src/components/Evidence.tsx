import { useEffect, useRef } from 'react';
import {
  Badge,
  EvidenceLinks,
  FactList,
  Section,
  anchor,
  date,
  decimal,
  humanize,
  money,
  narrative,
  rationaleText,
} from '../presentation';
import type { Evidence, ReviewView } from '../types';
import { useI18n } from '../i18n';
import { recommendationConcerns } from '../review/recommendationConcerns';
import { DamageComparison } from './DamageComparison';

export function KnowledgeEvidence({ view }: { view: ReviewView }) {
  const { locale, t, translate } = useI18n();
  return (
    <Section
      id="knowledge"
      title={t('Policy & Procedure Evidence')}
      eyebrow={t('06 / Grounded knowledge')}
      section={view.knowledge}
      help={t(
        'Policy clauses and internal procedures are distinct sources. An internal procedure is not a contractual clause.',
      )}
    >
      {(knowledge) => (
        <>
          {knowledge.insufficient_evidence && (
            <p className="notice">
              {t('Insufficient evidence — the recorded sources do not fully support an answer.')}
            </p>
          )}
          {knowledge.answer && (
            <div className="knowledge-answer">
              <h3>{t('Recorded answer')}</h3>
              <p>{translate(knowledge.answer)}</p>
              {locale === 'fr' && translate(knowledge.answer) !== knowledge.answer && (
                <details>
                  <summary>{t('Original answer text')}</summary>
                  <p lang="en">{knowledge.answer}</p>
                </details>
              )}
            </div>
          )}
          <div className="knowledge-sources">
            {knowledge.sources.map((source) => {
              const excerpt = source.text ? translate(source.text) : null;
              return (
                <article
                  className="knowledge-source"
                  id={anchor(`knowledge:${source.chunk_id}`)}
                  key={source.chunk_id}
                  tabIndex={-1}
                >
                  <div className="source-heading">
                    <Badge tone={source.source_type}>{t(source.source_type)}</Badge>
                    <span>
                      {source.source_type === 'PROCEDURE'
                        ? t('Internal guidance')
                        : source.source_type === 'POLICY'
                          ? t('Contractual source')
                          : t('Source type not recorded')}
                    </span>
                  </div>
                  <h3>{translate(source.title)}</h3>
                  <p className="source-section">{translate(source.section)}</p>
                  {excerpt && <blockquote>{excerpt}</blockquote>}
                  {locale === 'fr' && source.text && excerpt !== source.text && (
                    <details>
                      <summary>{t('Original source text')}</summary>
                      <blockquote lang="en">{source.text}</blockquote>
                    </details>
                  )}
                  <div className="source-footer">
                    <code>{source.chunk_id}</code>
                    <span>
                      {t('Similarity')} {decimal(source.similarity_score)}
                    </span>
                    {source.source_version && <span>v{source.source_version}</span>}
                  </div>
                </article>
              );
            })}
          </div>
          {!knowledge.sources.length && (
            <p className="muted">{t('No source chunks were retained in this result.')}</p>
          )}
        </>
      )}
    </Section>
  );
}

export function SimilarClaims({ view }: { view: ReviewView }) {
  const { t } = useI18n();
  return (
    <Section
      id="similar_claims"
      title={t('Similar Claims')}
      section={view.similar_claims}
      help={t(
        'Similarity is a retrieval metric for context. A similar claim does not establish wrongdoing.',
      )}
    >
      {(similar) =>
        !similar.results.length ? (
          <p className="muted">{t('No similar claims were returned in the recorded search.')}</p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th scope="col">{t('Claim')}</th>
                  <th scope="col">{t('Incident date')}</th>
                  <th scope="col">{t('Type')}</th>
                  <th scope="col">{t('Repair amount')}</th>
                  <th scope="col">{t('Similarity')}</th>
                </tr>
              </thead>
              <tbody>
                {similar.results.map((claim) => (
                  <tr key={claim.claim_id}>
                    <th scope="row">{claim.claim_id}</th>
                    <td>{date(claim.incident_date)}</td>
                    <td>{humanize(claim.claim_type)}</td>
                    <td>{money(claim.repair_amount, claim.repair_currency)}</td>
                    <td>{decimal(claim.similarity_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      }
    </Section>
  );
}

export function AgentRecommendation({
  view,
  onEvidence,
}: {
  view: ReviewView;
  onEvidence: (id: string) => void;
}) {
  const { locale, t, translate } = useI18n();
  const recommendation = view.agent.data?.recommendation;
  const { points, limits } = recommendationConcerns(recommendation?.uncertainties || []);
  const hasOriginalTranslation =
    locale === 'fr' &&
    recommendation &&
    (translate(recommendation.case_summary) !== recommendation.case_summary ||
      recommendation.rationale.some((item) => translate(item.statement) !== item.statement));
  return (
    <Section
      id="agent"
      title={t('Agent Recommendation')}
      eyebrow={t('07 / Advisory next step')}
      section={view.agent}
      className="agent-panel"
    >
      {(agent) =>
        agent.recommendation ? (
          <>
            <div className="agent-action">
              <span className="agent-action-icon" aria-hidden="true">
                ↗
              </span>
              <div>
                <p className="eyebrow">{t('Recommended next action')}</p>
                <h3>{humanize(agent.recommendation.recommended_next_action)}</h3>
              </div>
              <Badge tone="human">{t('HUMAN REVIEW REQUIRED')}</Badge>
            </div>
            <DamageComparison view={view} />
            <p className="case-summary">{narrative(agent.recommendation.case_summary)}</p>
            <div className="agent-columns">
              <div>
                <h3>{t('Evidence-grounded rationale')}</h3>
                <ol className="rationale">
                  {agent.recommendation.rationale.map((item, index) => (
                    <li key={index}>
                      <p>{rationaleText(item.statement, item.evidence_ids)}</p>
                      <EvidenceLinks ids={item.evidence_ids} onOpen={onEvidence} />
                    </li>
                  ))}
                </ol>
              </div>
              <div className="uncertainties">
                <section
                  id="agent-uncertainties"
                  className="agent-uncertainties"
                  aria-labelledby="agent-uncertainties-title"
                  tabIndex={-1}
                >
                  <div className="uncertainty-heading">
                    <h3 id="agent-uncertainties-title">{t('Points to verify')}</h3>
                    <span className="overview-count">{points.length}</span>
                  </div>
                  {points.length ? (
                    <ol className="uncertainty-list" aria-label={t('Points to verify')}>
                      {points.map((item, index) => (
                        <li key={index}>{translate(item)}</li>
                      ))}
                    </ol>
                  ) : (
                    <p>{t('No claim verification points recorded in this recommendation.')}</p>
                  )}
                </section>
                <details className="agent-analysis-details">
                  <summary>{t('Analysis details')}</summary>
                  {!!limits.length && (
                    <section
                      id="agent-analysis-limits"
                      className="agent-analysis-limits"
                      aria-labelledby="agent-analysis-limits-title"
                    >
                      <h3 id="agent-analysis-limits-title">{t('Analysis limits')}</h3>
                      <ul className="uncertainty-list" aria-label={t('Analysis limits')}>
                        {limits.map((item, index) => (
                          <li key={index}>{translate(item)}</li>
                        ))}
                      </ul>
                    </section>
                  )}
                  <h3>{t('Tools used')}</h3>
                  <ul className="tool-list">
                    {agent.tools_used.map((tool) => (
                      <li key={tool}>{humanize(tool)}</li>
                    ))}
                  </ul>
                  {!!agent.recommendation.uncertainties.length && (
                    <details className="recorded-uncertainties">
                      <summary>{t('Recorded uncertainty text')}</summary>
                      <ul>
                        {agent.recommendation.uncertainties.map((item, index) => (
                          <li
                            key={index}
                            lang={locale === 'fr' && translate(item) !== item ? 'en' : undefined}
                          >
                            {item}
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </details>
                <p className="human-field">
                  {t('Human review required:')}{' '}
                  <strong>
                    {agent.recommendation.human_review_required ? t('Yes') : t('Not recorded')}
                  </strong>
                </p>
              </div>
            </div>
            <div className="fact-block">
              <h3>{t('Evidence references')}</h3>
              <EvidenceLinks ids={agent.recommendation.evidence_ids} onOpen={onEvidence} />
            </div>
            {hasOriginalTranslation && (
              <details>
                <summary>{t('Original recommendation text')}</summary>
                <h3>{t('Case summary')}</h3>
                <p lang="en">{agent.recommendation.case_summary}</p>
                <h3>{t('Evidence-grounded rationale')}</h3>
                <ul>
                  {agent.recommendation.rationale.map((item, index) => (
                    <li key={index} lang="en">
                      {item.statement}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            <p className="model-note">
              {t('Agent {agent} · Prompt {prompt} · Run {run}', {
                agent: agent.agent_version,
                prompt: agent.prompt_version,
                run: agent.run_id,
              })}
            </p>
          </>
        ) : (
          <p className="muted">{t('No recommendation was recorded for this run.')}</p>
        )
      }
    </Section>
  );
}

export function Audit({
  view,
  onEvidence,
}: {
  view: ReviewView;
  onEvidence: (id: string) => void;
}) {
  const { t } = useI18n();
  return (
    <section id="audit" className="panel audit-panel" aria-labelledby="audit-title" tabIndex={-1}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{t('08 / Traceability')}</p>
          <h2 id="audit-title">{t('Audit / Provenance')}</h2>
        </div>
        <span className="muted">{t('Read-only record')}</span>
      </div>
      <details>
        <summary>{t('Technical details')}</summary>
        <div className="audit-grid">
          <div>
            <h3>{t('Model versions')}</h3>
            <FactList facts={view.audit.model_versions} />
          </div>
          <div>
            <h3>{t('Policy & prompt versions')}</h3>
            <FactList facts={view.audit.policy_versions} />
          </div>
          <div>
            <h3>{t('Recorded timestamps')}</h3>
            <FactList facts={view.audit.timestamps} />
          </div>
          <div>
            <h3>{t('Execution statuses')}</h3>
            <FactList facts={view.audit.execution_statuses} />
          </div>
        </div>
        <h3>{t('Source documents')}</h3>
        <ul className="source-document-list">
          {view.audit.source_documents.map((source) => (
            <li key={source}>{source}</li>
          ))}
        </ul>
        {!view.audit.source_documents.length && (
          <p className="muted">{t('No source documents recorded.')}</p>
        )}
        <h3>{t('Evidence IDs')}</h3>
        <EvidenceLinks ids={view.audit.evidence_ids} onOpen={onEvidence} />
        <p className="section-help">
          {t('Assembled {date}. Recommendations retain their evidence references.', {
            date: date(view.generated_at, true),
          })}
        </p>
      </details>
    </section>
  );
}

export function EvidenceDrawer({
  evidence,
  id,
  onClose,
}: {
  evidence: Evidence | undefined;
  id: string;
  onClose: () => void;
}) {
  const { t, translate } = useI18n();
  const closeButton = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeButton.current?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, [id]);
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside
        ref={panel}
        className="evidence-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="evidence-title"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === 'Escape') onClose();
          if (event.key === 'Tab') {
            const focusable = panel.current?.querySelectorAll<HTMLElement>(
              'button, a[href], [tabindex="0"]',
            );
            if (!focusable?.length) return;
            const first = focusable[0],
              last = focusable[focusable.length - 1];
            if (event.shiftKey && document.activeElement === first) {
              event.preventDefault();
              last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault();
              first.focus();
            }
          }
        }}
      >
        <div className="drawer-top">
          <span className="eyebrow">{t('Evidence source')}</span>
          <button
            className="icon-button"
            ref={closeButton}
            onClick={onClose}
            aria-label={t('Close evidence source')}
          >
            ×
          </button>
        </div>
        <h2 id="evidence-title">
          {evidence?.title ? translate(evidence.title) : t('Evidence reference')}
        </h2>
        <code className="drawer-id">{id}</code>
        {evidence ? (
          <>
            <p className="section-help">{evidence.source || t('Retained in the review record')}</p>
            <Badge>{humanize(evidence.section)}</Badge>
            {evidence.data ? (
              <FactList facts={evidence.data} />
            ) : (
              <p className="muted">
                {t(
                  'This reference identifies its source; no further structured fields were retained.',
                )}
              </p>
            )}
          </>
        ) : (
          <div className="empty-state">
            <p>
              {t(
                'The source details for this reference are not available in this stored review. The reference is retained for traceability.',
              )}
            </p>
          </div>
        )}
      </aside>
    </div>
  );
}
