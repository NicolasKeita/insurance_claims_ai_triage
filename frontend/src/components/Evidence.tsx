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
} from '../presentation';
import type { Evidence, ReviewView } from '../types';

export function KnowledgeEvidence({ view }: { view: ReviewView }) {
  return (
    <Section
      id="knowledge"
      title="Policy & Procedure Evidence"
      eyebrow="06 / Grounded knowledge"
      section={view.knowledge}
      help="Policy clauses and internal procedures are distinct sources. An internal procedure is not a contractual clause."
    >
      {(knowledge) => (
        <>
          {knowledge.insufficient_evidence && (
            <p className="notice">
              Insufficient evidence — the recorded sources do not fully support an answer.
            </p>
          )}
          {knowledge.answer && (
            <div className="knowledge-answer">
              <h3>Recorded answer</h3>
              <p>{knowledge.answer}</p>
            </div>
          )}
          <div className="knowledge-sources">
            {knowledge.sources.map((source) => (
              <article
                className="knowledge-source"
                id={anchor(`knowledge:${source.chunk_id}`)}
                key={source.chunk_id}
                tabIndex={-1}
              >
                <div className="source-heading">
                  <Badge tone={source.source_type}>{source.source_type}</Badge>
                  <span>
                    {source.source_type === 'PROCEDURE'
                      ? 'Internal guidance'
                      : source.source_type === 'POLICY'
                        ? 'Contractual source'
                        : 'Source type not recorded'}
                  </span>
                </div>
                <h3>{source.title}</h3>
                <p className="source-section">{source.section}</p>
                {source.text && <blockquote>{source.text}</blockquote>}
                <div className="source-footer">
                  <code>{source.chunk_id}</code>
                  <span>Similarity {decimal(source.similarity_score)}</span>
                  {source.source_version && <span>v{source.source_version}</span>}
                </div>
              </article>
            ))}
          </div>
          {!knowledge.sources.length && (
            <p className="muted">No source chunks were retained in this result.</p>
          )}
        </>
      )}
    </Section>
  );
}

export function SimilarClaims({ view }: { view: ReviewView }) {
  return (
    <Section
      id="similar_claims"
      title="Similar Claims"
      section={view.similar_claims}
      help="Similarity is a retrieval metric for context. A similar claim does not establish wrongdoing."
    >
      {(similar) =>
        !similar.results.length ? (
          <p className="muted">No similar claims were returned in the recorded search.</p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th scope="col">Claim</th>
                  <th scope="col">Incident date</th>
                  <th scope="col">Type</th>
                  <th scope="col">Repair amount</th>
                  <th scope="col">Similarity</th>
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
  return (
    <Section
      id="agent"
      title="Agent Recommendation"
      eyebrow="07 / Advisory next step"
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
                <p className="eyebrow">Recommended next action</p>
                <h3>{humanize(agent.recommendation.recommended_next_action)}</h3>
              </div>
              <Badge tone="human">HUMAN REVIEW REQUIRED</Badge>
            </div>
            <p className="case-summary">{agent.recommendation.case_summary}</p>
            <div className="agent-columns">
              <div>
                <h3>Evidence-grounded rationale</h3>
                <ol className="rationale">
                  {agent.recommendation.rationale.map((item, index) => (
                    <li key={index}>
                      <p>{item.statement}</p>
                      <EvidenceLinks ids={item.evidence_ids} onOpen={onEvidence} />
                    </li>
                  ))}
                </ol>
              </div>
              <div className="uncertainties">
                <h3>Uncertainties</h3>
                {agent.recommendation.uncertainties.length ? (
                  <ul>
                    {agent.recommendation.uncertainties.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p>No uncertainties recorded by this run.</p>
                )}
                <h3>Tools used</h3>
                <ul className="tool-list">
                  {agent.tools_used.map((tool) => (
                    <li key={tool}>{humanize(tool)}</li>
                  ))}
                </ul>
                <p className="human-field">
                  Human review required:{' '}
                  <strong>
                    {agent.recommendation.human_review_required ? 'Yes' : 'Not recorded'}
                  </strong>
                </p>
              </div>
            </div>
            <div className="fact-block">
              <h3>Evidence references</h3>
              <EvidenceLinks ids={agent.recommendation.evidence_ids} onOpen={onEvidence} />
            </div>
            <p className="model-note">
              Agent {agent.agent_version} · Prompt {agent.prompt_version} · Run {agent.run_id}
            </p>
          </>
        ) : (
          <p className="muted">No recommendation was recorded for this run.</p>
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
  return (
    <section id="audit" className="panel audit-panel" aria-labelledby="audit-title" tabIndex={-1}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">08 / Traceability</p>
          <h2 id="audit-title">Audit / Provenance</h2>
        </div>
        <span className="muted">Read-only record</span>
      </div>
      <details>
        <summary>Technical details</summary>
        <div className="audit-grid">
          <div>
            <h3>Model versions</h3>
            <FactList facts={view.audit.model_versions} />
          </div>
          <div>
            <h3>Policy & prompt versions</h3>
            <FactList facts={view.audit.policy_versions} />
          </div>
          <div>
            <h3>Recorded timestamps</h3>
            <FactList facts={view.audit.timestamps} />
          </div>
          <div>
            <h3>Execution statuses</h3>
            <FactList facts={view.audit.execution_statuses} />
          </div>
        </div>
        <h3>Source documents</h3>
        <ul className="source-document-list">
          {view.audit.source_documents.map((source) => (
            <li key={source}>{source}</li>
          ))}
        </ul>
        {!view.audit.source_documents.length && (
          <p className="muted">No source documents recorded.</p>
        )}
        <h3>Evidence IDs</h3>
        <EvidenceLinks ids={view.audit.evidence_ids} onOpen={onEvidence} />
        <p className="section-help">
          Assembled {date(view.generated_at, true)}. Recommendations retain their evidence
          references.
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
          <span className="eyebrow">Evidence source</span>
          <button
            className="icon-button"
            ref={closeButton}
            onClick={onClose}
            aria-label="Close evidence source"
          >
            ×
          </button>
        </div>
        <h2 id="evidence-title">{evidence?.title || 'Evidence reference'}</h2>
        <code className="drawer-id">{id}</code>
        {evidence ? (
          <>
            <p className="section-help">{evidence.source || 'Retained in the review record'}</p>
            <Badge>{humanize(evidence.section)}</Badge>
            {evidence.data ? (
              <FactList facts={evidence.data} />
            ) : (
              <p className="muted">
                This reference identifies its source; no further structured fields were retained.
              </p>
            )}
          </>
        ) : (
          <div className="empty-state">
            <p>
              The source details for this reference are not available in this stored review. The
              reference is retained for traceability.
            </p>
          </div>
        )}
      </aside>
    </div>
  );
}
