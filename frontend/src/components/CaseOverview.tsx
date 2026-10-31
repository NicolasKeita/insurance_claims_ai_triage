import { useState } from 'react';
import { Badge, StateBadge, humanize } from '../presentation';
import type { ReviewView, VisionImage } from '../types';

function PhotoSummary({
  images,
  listedCount,
  status,
  onOpen,
}: {
  images: VisionImage[];
  listedCount: number;
  status: ReviewView['vision']['status'];
  onOpen: () => void;
}) {
  const [failed, setFailed] = useState<string[]>([]);
  const photographs = images.filter(
    (image) => image.image_url?.startsWith('/v1/') && !failed.includes(image.image_url),
  );
  const preview = photographs[0];
  return (
    <article className="overview-evidence-card overview-photo-card">
      <div className="overview-evidence-copy">
        <h3>Photographs</h3>
        <p className="overview-evidence-count">
          {photographs.length
            ? `${photographs.length} source ${photographs.length === 1 ? 'photo' : 'photos'}`
            : listedCount || images.length
              ? 'Photos unavailable'
              : 'No photographs listed'}
        </p>
        <p className="overview-evidence-note">
          {status === 'AVAILABLE'
            ? 'Recorded vision results available.'
            : status === 'NO_EVIDENCE'
              ? 'No supported observations recorded.'
              : `Vision ${humanize(status).toLowerCase()}.`}
        </p>
        <button type="button" className="overview-link" onClick={onOpen}>
          View photographs <span aria-hidden="true">↗</span>
        </button>
      </div>
      {preview?.image_url && (
        <img
          className="overview-photo-preview"
          src={preview.image_url}
          alt="Source claim photograph preview"
          onError={() => setFailed((current) => [...current, preview.image_url!])}
        />
      )}
    </article>
  );
}

export function CaseOverview({
  view,
  onOpenDetails,
}: {
  view: ReviewView;
  onOpenDetails: (section: string) => void;
}) {
  const agent =
    view.agent.status === 'AVAILABLE' && view.agent.data?.status === 'COMPLETED'
      ? view.agent.data
      : null;
  const recommendation = agent?.recommendation;
  const investigation = view.investigation.status === 'AVAILABLE' ? view.investigation.data : null;
  const signals = investigation?.signals || [];
  const documents = view.documents.data?.items || view.claim.documents;
  const availableDocuments = documents.filter((document) => document.available).length;
  const missingDocuments = documents.filter((document) => !document.available).length;
  const uncertainties = recommendation?.uncertainties || [];

  return (
    <div className="case-overview">
      <section className="overview-next-step" aria-labelledby="overview-next-step-title">
        <div className="overview-next-step-symbol" aria-hidden="true">
          ↗
        </div>
        <div className="overview-next-step-copy">
          <p className="eyebrow">Suggested next step</p>
          <h2 id="overview-next-step-title">
            {recommendation
              ? humanize(recommendation.recommended_next_action)
              : 'No saved recommendation'}
          </h2>
          <p>
            {recommendation
              ? 'Saved agent guidance. Open the full recommendation for its rationale and sources.'
              : view.agent.reason ||
                (view.agent.status === 'FAILED'
                  ? 'The agent run failed. Its recorded status is available in the details.'
                  : 'An agent recommendation has not been recorded for this review.')}
          </p>
        </div>
        {recommendation ? (
          <button type="button" className="overview-link" onClick={() => onOpenDetails('agent')}>
            Read full recommendation <span aria-hidden="true">↗</span>
          </button>
        ) : (
          <StateBadge
            status={view.agent.status === 'AVAILABLE' ? 'NO_EVIDENCE' : view.agent.status}
          />
        )}
      </section>

      <div className="overview-grid">
        <section className="overview-attention" aria-labelledby="overview-attention-title">
          <div className="overview-card-heading">
            <h2 id="overview-attention-title">What needs attention</h2>
            {investigation ? (
              <span className="overview-count">
                {signals.length} recorded {signals.length === 1 ? 'signal' : 'signals'}
              </span>
            ) : (
              <StateBadge status={view.investigation.status} />
            )}
          </div>
          {investigation ? (
            <>
              {signals.length ? (
                <ul className="overview-signal-list">
                  {signals.slice(0, 3).map((signal) => (
                    <li key={signal.code}>
                      <div>
                        <h3>{signal.title || humanize(signal.code)}</h3>
                        <Badge tone={signal.severity}>{humanize(signal.severity)}</Badge>
                      </div>
                      <p>{signal.explanation}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="overview-empty">No review signals in this saved assessment.</p>
              )}
              {investigation.partial && (
                <p className="overview-scope">
                  <strong>Partial assessment. </strong>
                  {view.investigation.reason ||
                    (investigation.unavailable_inputs?.length
                      ? `Checks not run: ${investigation.unavailable_inputs.map(humanize).join(', ')}.`
                      : 'This saved assessment covers a limited set of checks.')}
                </p>
              )}
            </>
          ) : (
            <p className="overview-empty">
              {view.investigation.reason ||
                (view.investigation.status === 'FAILED'
                  ? 'The investigation assessment failed.'
                  : 'No investigation assessment is available. Missing results do not mean there are no review signals.')}
            </p>
          )}
          <div className="overview-attention-footer">
            <button
              type="button"
              className="overview-link"
              onClick={() => onOpenDetails('investigation')}
            >
              {signals.length > 3 ? `View all ${signals.length} signals` : 'View all signals'}{' '}
              <span aria-hidden="true">↗</span>
            </button>
          </div>
        </section>

        <section className="overview-evidence-summary" aria-label="Available case evidence">
          <article className="overview-evidence-card">
            <div className="overview-evidence-copy">
              <h3>Documents</h3>
              <p className="overview-evidence-count">
                {documents.length
                  ? `${availableDocuments} available · ${missingDocuments} missing`
                  : 'No documents listed'}
              </p>
              <p className="overview-evidence-note">
                Availability from the case document inventory.
              </p>
              <button
                type="button"
                className="overview-link"
                onClick={() => onOpenDetails('documents')}
              >
                Review documents <span aria-hidden="true">↗</span>
              </button>
            </div>
            <span className="overview-document-symbol" aria-hidden="true">
              ▤
            </span>
          </article>
          <PhotoSummary
            key={view.claim.claim_id}
            images={view.vision.data?.images || []}
            listedCount={view.claim.images.length}
            status={view.vision.status}
            onOpen={() => onOpenDetails('vision')}
          />
        </section>
      </div>

      {!!uncertainties.length && (
        <section className="overview-uncertainties" aria-labelledby="overview-uncertainties-title">
          <div>
            <h2 id="overview-uncertainties-title">Limits of this recommendation</h2>
            <p>
              {uncertainties.length} recorded{' '}
              {uncertainties.length === 1 ? 'uncertainty' : 'uncertainties'}. Review{' '}
              {uncertainties.length === 1 ? 'it' : 'them'} before making a decision.
            </p>
          </div>
          <button type="button" className="overview-link" onClick={() => onOpenDetails('agent')}>
            View uncertainties <span aria-hidden="true">↗</span>
          </button>
        </section>
      )}
    </div>
  );
}
