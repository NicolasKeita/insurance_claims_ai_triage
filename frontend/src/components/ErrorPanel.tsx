import type { RequestFailure } from '../request';

const backendCommand = String.raw`$env:CLAIMS_REVIEW_DEMO = '1'
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000`;

export function ErrorPanel({ error, onRetry }: { error: RequestFailure; onRetry: () => void }) {
  const offline = error.kind === 'backend-unavailable';
  const heading = {
    'backend-unavailable': 'Backend unavailable',
    'claim-not-found': 'Claim not found',
    'service-unavailable': 'Review service unavailable',
    'request-failed': 'The request could not be completed',
  }[error.kind];

  return (
    <section className="panel request-error" role="alert" aria-labelledby="request-error-title">
      <p className="eyebrow">{offline ? 'Backend connection' : 'Review unavailable'}</p>
      <h1 id="request-error-title" tabIndex={-1}>
        {heading}
      </h1>
      {offline ? (
        <>
          <p>
            The console cannot reach FastAPI at <code>http://127.0.0.1:8000</code>. Start the
            backend, then retry the connection.
          </p>
          <div className="backend-start-help">
            <h2>Start the prepared demo backend</h2>
            <p>
              Run these commands in PowerShell from the repository root. Keep that window open while
              using the console.
            </p>
            <pre aria-label="PowerShell commands to start the prepared demo backend">
              <code>{backendCommand}</code>
            </pre>
            <p>This reads your prepared demo snapshots; it does not run models or the benchmark.</p>
            <p className="docker-help">
              PostgreSQL and Qdrant are not needed for the prepared demo. Docker starts those
              services separately; it does not start FastAPI.
            </p>
          </div>
        </>
      ) : error.kind === 'claim-not-found' ? (
        <p>This claim was not found. Check the claim ID or choose an available demo case.</p>
      ) : error.kind === 'service-unavailable' ? (
        <>
          <p>
            FastAPI is responding, but a required service or configuration is unavailable. Check the
            backend terminal for details, then try again.
          </p>
          <p className="service-help">
            In database mode, check PostgreSQL and DATABASE_URL. In demo mode, check that the demo
            snapshots have been prepared.
          </p>
        </>
      ) : (
        <p>
          {error.resource === 'catalog' ? 'The demo list' : 'The review'} could not be loaded
          {error.status && error.status >= 400 ? ` (HTTP ${error.status})` : ''}. The backend
          returned an error or an unreadable response. Check its terminal for details and try again.
        </p>
      )}
      <button className="primary-button" type="button" onClick={onRetry}>
        {offline ? 'Retry connection' : 'Try again'}
      </button>
      <p className="retry-help">
        Retry refreshes the demo list and reopens the selected claim when available.
      </p>
    </section>
  );
}
