import { useI18n } from '../i18n';
import type { RequestFailure } from '../request';
import { isStaticDemo } from '../assets';

const backendCommand = String.raw`$env:CLAIMS_REVIEW_DEMO = '1'
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000`;
const dockerCommand = `docker compose up --build -d frontend_demo
docker compose logs api`;

export function ErrorPanel({ error, onRetry }: { error: RequestFailure; onRetry: () => void }) {
  const { t } = useI18n();

  if (isStaticDemo()) return <StaticErrorPanel error={error} onRetry={onRetry} />;

  const docker = process.env.NEXT_PUBLIC_REVIEW_RUNTIME === 'docker';
  const offline = error.kind === 'backend-unavailable';
  const heading = {
    'backend-unavailable': 'Backend unavailable',
    'claim-not-found': 'Claim not found',
    'service-unavailable': 'Review service unavailable',
    'request-failed': 'The request could not be completed',
  }[error.kind];

  return (
    <section className="panel request-error" role="alert" aria-labelledby="request-error-title">
      <p className="eyebrow">{t(offline ? 'Backend connection' : 'Review unavailable')}</p>
      <h1 id="request-error-title" tabIndex={-1}>
        {t(heading)}
      </h1>
      {offline ? (
        <>
          <p>
            {docker ? (
              t(
                'The console cannot reach the API container. Start the Docker Compose demo, then retry the connection.',
              )
            ) : (
              <>
                {t('The console cannot reach FastAPI at')} <code>http://127.0.0.1:8000</code>
                {t('. Start the backend, then retry the connection.')}
              </>
            )}
          </p>
          {docker ? (
            <DockerStartHelp />
          ) : (
            <div className="backend-start-help">
              <h2>{t('Start the prepared demo backend')}</h2>
              <p>
                {t(
                  'Run these commands in PowerShell from the repository root. Keep that window open while using the console.',
                )}
              </p>
              <pre aria-label={t('PowerShell commands to start the prepared demo backend')}>
                <code>{backendCommand}</code>
              </pre>
              <p>
                {t(
                  'This reads your prepared demo snapshots; it does not run models or the benchmark.',
                )}
              </p>
              <p className="docker-help">
                {t(
                  'PostgreSQL and Qdrant are not needed for the prepared demo. The Docker console starts FastAPI and the frontend together.',
                )}
              </p>
            </div>
          )}
        </>
      ) : error.kind === 'claim-not-found' ? (
        <p>{t('This claim was not found. Check the claim ID or choose an available demo case.')}</p>
      ) : error.kind === 'service-unavailable' ? (
        docker ? (
          <>
            <p>
              {t(
                'The API container is responding, but its analysis archive is unavailable. Check analysis_results/manifest.json and the API logs.',
              )}
            </p>
            <pre>
              <code>docker compose logs api</code>
            </pre>
          </>
        ) : (
          <>
            <p>
              {t(
                'FastAPI is responding, but a required service or configuration is unavailable. Check the backend terminal for details, then try again.',
              )}
            </p>
            <p className="service-help">
              {t(
                'In database mode, check PostgreSQL and DATABASE_URL. In demo mode, check that the demo snapshots have been prepared.',
              )}
            </p>
          </>
        )
      ) : (
        <p>
          {t(
            '{resource} could not be loaded{status}. The backend returned an error or an unreadable response. Check its terminal for details and try again.',
            {
              resource: t(error.resource === 'catalog' ? 'The demo list' : 'The review'),
              status: error.status && error.status >= 400 ? ` (HTTP ${error.status})` : '',
            },
          )}
        </p>
      )}
      <button className="primary-button" type="button" onClick={onRetry}>
        {t(offline ? 'Retry connection' : 'Try again')}
      </button>
      <p className="retry-help">
        {t('Retry refreshes the demo list and reopens the selected claim when available.')}
      </p>
    </section>
  );
}

function StaticErrorPanel({ error, onRetry }: { error: RequestFailure; onRetry: () => void }) {
  const { t } = useI18n();
  const missing = error.kind === 'claim-not-found';
  return (
    <section className="panel request-error" role="alert" aria-labelledby="request-error-title">
      <p className="eyebrow">{t('Review unavailable')}</p>
      <h1 id="request-error-title" tabIndex={-1}>
        {t(missing ? 'Claim not found' : 'Demo unavailable')}
      </h1>
      <p>
        {t(
          missing
            ? 'The selected analysis is not included in this demo. Check the link or choose an available case.'
            : 'The prepared demo files could not be loaded. Reload the page or try again.',
        )}
      </p>
      <button className="primary-button" type="button" onClick={onRetry}>
        {t('Try again')}
      </button>
      <p className="retry-help">
        {t('Retry refreshes the demo list and reopens the selected claim when available.')}
      </p>
    </section>
  );
}

function DockerStartHelp() {
  const { t } = useI18n();
  return (
    <div className="backend-start-help">
      <h2>{t('Start the Docker demo')}</h2>
      <p>{t('Run these commands from the repository root with Docker Desktop running.')}</p>
      <pre aria-label={t('Start the Docker demo')}>
        <code>{dockerCommand}</code>
      </pre>
      <p>
        {t(
          'The analysis_results folder must contain exported analyses. Inspect the API logs if the container does not start.',
        )}
      </p>
    </div>
  );
}
