import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { Documents, DocumentConsistency } from './components/Documents';
import {
  AgentRecommendation,
  Audit,
  EvidenceDrawer,
  KnowledgeEvidence,
  SimilarClaims,
} from './components/Evidence';
import { Anomaly, ClaimHistory, Investigation, Triage } from './components/Intelligence';
import { VisualConsistency, VisualEvidence } from './components/Vision';
import { ErrorPanel } from './components/ErrorPanel';
import { Badge, Metric, anchor, date, humanize, money } from './presentation';
import { fetchJson, isAbortError, requestFailure } from './request';
import type { RequestFailure } from './request';
import type { DemoClaim, ReviewView } from './types';

const navigation = [
  ['documents', 'Documents'],
  ['consistency', 'Document consistency'],
  ['investigation', 'Investigation signals'],
  ['triage', 'Triage & anomaly'],
  ['history', 'Historical context'],
  ['vision', 'Visual evidence'],
  ['visual_consistency', 'Visual consistency'],
  ['knowledge', 'Policy & procedure'],
  ['similar_claims', 'Similar claims'],
  ['agent', 'Agent recommendation'],
  ['audit', 'Audit / provenance'],
] as const;

export function ReviewConsole({ view }: { view: ReviewView }) {
  const [evidenceId, setEvidenceId] = useState<string | null>(null);
  const claim = view.claim;
  const investigation = view.investigation.status === 'AVAILABLE' ? view.investigation.data : null;
  function openEvidence(id: string) {
    const source = view.evidence.find((item) => item.id === id);
    const target =
      document.getElementById(anchor(id)) || (source && document.getElementById(source.section));
    if (target) {
      target.scrollIntoView?.({ behavior: 'instant', block: 'center' });
      target.classList.add('evidence-highlight');
      target.focus({ preventScroll: true });
      window.setTimeout(() => target.classList.remove('evidence-highlight'), 4000);
    }
    setEvidenceId(id);
  }
  return (
    <>
      <header className="claim-heading">
        <div className="claim-title">
          <p className="eyebrow">Case workspace / Claim Review</p>
          <div>
            <h1>{claim.claim_id}</h1>
            <Badge tone="status">{humanize(claim.status)}</Badge>
            {view.mode === 'demo' && <Badge tone="demo">Controlled demo</Badge>}
          </div>
          <p>
            {humanize(claim.claim_type)} <span aria-hidden="true">·</span> {claim.policy.product}{' '}
            <span aria-hidden="true">·</span> {claim.customer.customer_id}
          </p>
        </div>
        <div className="priority-card">
          <p className="eyebrow">Overall review priority</p>
          {investigation ? (
            <>
              <div>
                <span
                  className={`priority-dot ${investigation.review_priority.toLowerCase()}`}
                  aria-hidden="true"
                />
                <strong>{investigation.review_priority}</strong>
                <span className="review-score">
                  {investigation.review_score}
                  <small> / 100</small>
                </span>
              </div>
              <p>Review score · Policy {investigation.policy_version}</p>
            </>
          ) : (
            <>
              <strong className="priority-unavailable">Not available</strong>
              <p>{view.investigation.reason || 'Investigation has not been recorded.'}</p>
            </>
          )}
          <p className="priority-disclaimer">Review prioritization — not a fraud probability.</p>
        </div>
      </header>
      <div className="human-banner">
        <span className="human-banner-symbol" aria-hidden="true">
          ◇
        </span>
        <strong>HUMAN REVIEW REQUIRED</strong>
        <span>AI-assisted guidance. Final decisions remain with the claims handler.</span>
      </div>
      <dl className="case-metrics">
        <Metric label="Incident date" value={date(claim.incident.date)} />
        <Metric label="Vehicle" value={`${claim.vehicle.make} ${claim.vehicle.model}`} />
        <Metric
          label="Repair amount"
          value={money(claim.repair_estimate.amount, claim.repair_estimate.currency)}
        />
        <Metric label="Incident location" value={claim.incident.location} />
      </dl>
      <div className="case-context">
        <span>{claim.vehicle.year} vehicle</span>
        <span>{humanize(claim.incident.collision_type)}</span>
        <span>Injuries declared: {claim.incident.injuries_declared ? 'Yes' : 'No'}</span>
        <span>Policy {claim.policy.policy_id}</span>
      </div>
      <div className="review-grid">
        <div className="document-column">
          <Documents view={view} />
          <DocumentConsistency view={view} />
        </div>
        <Investigation view={view} onEvidence={openEvidence} />
      </div>
      <div className="review-grid equal">
        <Triage view={view} />
        <Anomaly view={view} />
      </div>
      <ClaimHistory view={view} />
      <VisualEvidence view={view} />
      <VisualConsistency view={view} />
      <KnowledgeEvidence view={view} />
      <SimilarClaims view={view} />
      <AgentRecommendation view={view} onEvidence={openEvidence} />
      <Audit view={view} onEvidence={openEvidence} />
      <footer className="review-footer">
        <span>Insurance AI · Claim review console</span>
        <span>Results retain their recorded execution state and provenance.</span>
      </footer>
      {evidenceId && (
        <EvidenceDrawer
          id={evidenceId}
          evidence={view.evidence.find((item) => item.id === evidenceId)}
          onClose={() => setEvidenceId(null)}
        />
      )}
    </>
  );
}

export default function App() {
  const [demoClaims, setDemoClaims] = useState<DemoClaim[]>([]);
  const [view, setView] = useState<ReviewView | null>(null);
  const [claimId, setClaimId] = useState('');
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState<RequestFailure | null>(null);
  const [demoUnavailable, setDemoUnavailable] = useState(false);
  const pending = useRef<AbortController | null>(null);
  const catalogPending = useRef<AbortController | null>(null);

  const loadClaim = useCallback(async (id: string) => {
    const trimmed = id.trim();
    if (!trimmed) return;
    pending.current?.abort();
    const request = new AbortController();
    pending.current = request;
    setClaimId(trimmed);
    setError(null);
    setLoading(true);
    try {
      const result = await fetchJson<ReviewView>(
        `/v1/claims/${encodeURIComponent(trimmed)}/review`,
        { signal: request.signal, resource: 'claim' },
      );
      if (!request.signal.aborted) {
        setView(result);
        const url = new URL(window.location.href);
        url.searchParams.set('claim', trimmed);
        window.history.replaceState(null, '', url);
      }
    } catch (failure) {
      if (!request.signal.aborted && !isAbortError(failure)) {
        setView(null);
        setError(requestFailure(failure));
      }
    } finally {
      if (!request.signal.aborted) setLoading(false);
    }
  }, []);

  const loadWorkspace = useCallback(
    async (selectedId = '') => {
      catalogPending.current?.abort();
      pending.current?.abort();
      const controller = new AbortController();
      catalogPending.current = controller;
      setInitializing(true);
      setLoading(false);
      setError(null);
      let claims: DemoClaim[] = [];
      let catalogError: RequestFailure | null = null;
      try {
        claims = await fetchJson<DemoClaim[]>('/v1/review/demo-claims', {
          signal: controller.signal,
          resource: 'catalog',
        });
        if (!controller.signal.aborted) {
          setDemoClaims(claims);
          setDemoUnavailable(false);
        }
      } catch (failure) {
        if (controller.signal.aborted) return;
        if (isAbortError(failure)) {
          setInitializing(false);
          return;
        }
        setDemoClaims([]);
        setDemoUnavailable(true);
        catalogError = requestFailure(failure);
      }
      if (controller.signal.aborted) return;
      setInitializing(false);
      const initial = selectedId.trim() || claims[0]?.claim_id;
      if (initial) await loadClaim(initial);
      else {
        setView(null);
        setError(catalogError);
      }
    },
    [loadClaim],
  );

  useEffect(() => {
    const initial = new URLSearchParams(window.location.search).get('claim') || '';
    void loadWorkspace(initial);
    return () => {
      catalogPending.current?.abort();
      pending.current?.abort();
    };
  }, [loadWorkspace]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void loadClaim(claimId);
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to claim review
      </a>
      <aside className="sidebar" aria-label="Review workspace">
        <a className="brand" href="/" aria-label="Insurance AI home">
          <span className="brand-mark" aria-hidden="true">
            ▦
          </span>
          <span>
            INSURANCE<span>AI REVIEW</span>
          </span>
        </a>
        <div className="workspace-label">
          <span className="workspace-dot" aria-hidden="true" />
          Claims workspace
        </div>
        <div className="sidebar-heading">REVIEW SECTIONS</div>
        <nav aria-label="Claim sections">
          {navigation.map(([id, label], index) => (
            <a key={id} href={`#${id}`}>
              <span className="nav-number" aria-hidden="true">
                {String(index + 1).padStart(2, '0')}
              </span>
              {label}
            </a>
          ))}
        </nav>
        <div className="demo-cases">
          <div className="sidebar-heading">CONTROLLED DEMO CASES</div>
          {demoClaims.map((claim) => (
            <button
              type="button"
              key={claim.claim_id}
              className={
                view?.claim.claim_id === claim.claim_id ? 'demo-case selected' : 'demo-case'
              }
              onClick={() => void loadClaim(claim.claim_id)}
              disabled={loading || initializing}
            >
              <span className="demo-case-indicator" aria-hidden="true" />
              <span>
                <strong>{claim.title}</strong>
                <small>{claim.description}</small>
              </span>
            </button>
          ))}
          {!demoClaims.length && !initializing && (
            <p className="sidebar-note">
              {demoUnavailable
                ? 'Demo list unavailable. You can open a claim by ID.'
                : 'Use a persisted claim ID, or enable the minimal demo backend.'}
            </p>
          )}
        </div>
        <div className="sidebar-footer">
          <span className="advisory-icon" aria-hidden="true">
            ◇
          </span>
          <div>
            <strong>Advisory by design</strong>
            <p>Evidence for human decisions.</p>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <div className="topbar">
          <span>
            Claim Review Console{' '}
            <span className="topbar-divider" aria-hidden="true">
              /
            </span>{' '}
            Portfolio workspace
          </span>
          <span className="workspace-status">
            <span aria-hidden="true" />
            Read-only review
          </span>
        </div>
        <div className="claim-toolbar">
          <form onSubmit={submit}>
            <label htmlFor="claim-id">Open claim</label>
            <input
              id="claim-id"
              name="claim-id"
              type="text"
              value={claimId}
              onChange={(event) => setClaimId(event.target.value)}
              placeholder="Enter a claim ID"
              autoComplete="off"
              required
            />
            <button
              className="primary-button"
              type="submit"
              disabled={loading || initializing || !claimId.trim()}
            >
              Open review <span aria-hidden="true">↗</span>
            </button>
          </form>
          {view && (
            <button
              className="text-button"
              type="button"
              disabled={loading || initializing}
              onClick={() => void loadClaim(view.claim.claim_id)}
            >
              <span aria-hidden="true">↻</span> Refresh stored results
            </button>
          )}
        </div>
        <main id="main-content" tabIndex={-1} aria-busy={loading || initializing}>
          {loading || initializing ? (
            <div className="loading" role="status">
              <span className="loading-mark" aria-hidden="true" />
              <h1>Loading claim review</h1>
              <p>Reading available evidence and recorded assessments…</p>
            </div>
          ) : error ? (
            <ErrorPanel error={error} onRetry={() => void loadWorkspace(claimId)} />
          ) : view ? (
            <ReviewConsole key={view.claim.claim_id} view={view} />
          ) : (
            <section className="welcome">
              <p className="eyebrow">Evidence-grounded review</p>
              <h1>
                Every claim has a story.
                <br />
                Start with the evidence.
              </h1>
              <p>
                Open a claim by ID or select a controlled demo case to explore its documents,
                investigation signals and recorded assessments.
              </p>
              <div className="welcome-note">
                Human judgment remains at the centre of every decision.
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
