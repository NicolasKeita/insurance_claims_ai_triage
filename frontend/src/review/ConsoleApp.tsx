import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ErrorPanel } from '../components/ErrorPanel';
import { ThemeToggle } from '../components/ThemeToggle';
import { fetchJson, isAbortError, requestFailure } from '../request';
import type { RequestFailure } from '../request';
import type { DemoClaim, ReviewView } from '../types';
import { ReviewConsole } from './ReviewWorkspace';
import { ScenarioPicker } from './ScenarioPicker';
import { scenarioCopy } from './scenarios';

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
        {
          signal: request.signal,
          resource: 'claim',
        },
      );
      if (!request.signal.aborted) {
        setView(result);
        const url = new URL(window.location.href);
        url.searchParams.set('claim', trimmed);
        url.hash = '';
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
      let catalogError: RequestFailure | null = null;
      try {
        const claims = await fetchJson<DemoClaim[]>('/v1/review/demo-claims', {
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
      if (selectedId.trim()) await loadClaim(selectedId);
      else {
        setView(null);
        setError(catalogError);
      }
    },
    [loadClaim],
  );

  useEffect(() => {
    void loadWorkspace(new URLSearchParams(window.location.search).get('claim') || '');
    return () => {
      catalogPending.current?.abort();
      pending.current?.abort();
    };
  }, [loadWorkspace]);

  function backToCases() {
    pending.current?.abort();
    setView(null);
    setError(null);
    setClaimId('');
    setLoading(false);
    const url = new URL(window.location.href);
    url.searchParams.delete('claim');
    url.hash = '';
    window.history.replaceState(null, '', url);
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void loadClaim(claimId);
  }

  const selectedScenario =
    view && demoClaims.find((claim) => claim.claim_id === view.claim.claim_id);
  const busy = loading || initializing;

  useEffect(() => {
    if (busy) return;
    const target = document.getElementById(
      error ? 'request-error-title' : view ? 'review-case-title' : 'journey-title',
    );
    target?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [busy, view, error]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to claim review
      </a>
      <aside className="sidebar guided-sidebar" aria-label="Review workspace">
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
          Interactive review
        </div>
        {(view || error) && demoClaims.length > 0 ? (
          <>
            <div className="sidebar-heading">PREPARED CASES</div>
            <ScenarioPicker
              claims={demoClaims}
              compact
              selectedId={view?.claim.claim_id}
              disabled={busy}
              onSelect={(id) => void loadClaim(id)}
            />
          </>
        ) : (
          <p className="sidebar-home-note">
            {demoUnavailable
              ? error?.kind === 'backend-unavailable'
                ? 'Start FastAPI to load the prepared cases.'
                : 'The case list could not be loaded. Check the connection details.'
              : demoClaims.length > 0
                ? 'Choose a case to begin. You do not need a claim ID.'
                : 'Open a stored claim using its claim ID.'}
          </p>
        )}
        <div className="sidebar-footer">
          <span className="advisory-icon" aria-hidden="true">
            ◇
          </span>
          <div>
            <strong>Human decisions</strong>
            <p>Evidence you can trace.</p>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <div className="topbar">
          <span>Claim review</span>
          <div className="topbar-actions">
            <span className="workspace-status">Prepared evidence</span>
            <ThemeToggle />
          </div>
        </div>
        <main id="main-content" className="guided-content" tabIndex={-1} aria-busy={busy}>
          {busy ? (
            <div className="loading" role="status">
              <span className="loading-mark" aria-hidden="true" />
              <h1>Loading claim review</h1>
              <p>Reading the saved evidence…</p>
            </div>
          ) : error ? (
            <>
              <ErrorPanel error={error} onRetry={() => void loadWorkspace(claimId)} />
              {demoClaims.length > 0 && (
                <button
                  className="case-breadcrumb"
                  type="button"
                  aria-label="Back to demo cases"
                  onClick={backToCases}
                >
                  ← Back to demo cases
                </button>
              )}
            </>
          ) : view ? (
            <ReviewConsole
              key={view.claim.claim_id}
              view={view}
              scenarioTitle={selectedScenario ? scenarioCopy(selectedScenario).title : undefined}
              onBack={backToCases}
            />
          ) : (
            <section className="journey-home">
              <div className="journey-intro">
                <p className="eyebrow">Explore the platform</p>
                <h1 id="journey-title" tabIndex={-1}>
                  Explore a claim review
                </h1>
                <p>
                  {demoClaims.length > 0
                    ? 'Choose a prepared case. Understand the findings, then follow the evidence behind them.'
                    : 'Open a stored claim by ID to explore its recorded findings and supporting evidence.'}
                </p>
              </div>
              <ol className="journey-steps" aria-label="How to explore the demo">
                <li>
                  <span>1</span>Choose a case
                </li>
                <li>
                  <span>2</span>Read the overview
                </li>
                <li>
                  <span>3</span>Inspect the evidence
                </li>
              </ol>
              {demoClaims.length > 0 && (
                <ScenarioPicker claims={demoClaims} onSelect={(id) => void loadClaim(id)} />
              )}
              <details className="demo-about">
                <summary>About this demo</summary>
                <p>
                  These controlled, synthetic cases show how document analysis, ML, computer vision
                  and source-grounded AI support a human reviewer. Results are already prepared.
                  Some components may be unavailable; those limits remain visible. No insurance
                  decision is made automatically.
                </p>
              </details>
            </section>
          )}
          {!busy && (
            <details className="manual-claim" open={demoClaims.length === 0 || undefined}>
              <summary>Open a claim by ID</summary>
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
                  <button className="primary-button" type="submit" disabled={!claimId.trim()}>
                    Open review <span aria-hidden="true">↗</span>
                  </button>
                </form>
                {view && (
                  <button
                    className="text-button"
                    type="button"
                    onClick={() => void loadClaim(view.claim.claim_id)}
                  >
                    Refresh stored results
                  </button>
                )}
              </div>
            </details>
          )}
        </main>
      </div>
    </div>
  );
}
