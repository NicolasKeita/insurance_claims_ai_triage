import { useState } from 'react';
import { Badge, StateBadge, humanize } from '../presentation';
import type { ReviewView, VisionImage } from '../types';

function PhotoFeature({
  images,
  listedCount,
  status,
  mode,
  onOpen,
}: {
  images: VisionImage[];
  listedCount: number;
  status: ReviewView['vision']['status'];
  mode: ReviewView['mode'];
  onOpen: () => void;
}) {
  const [failed, setFailed] = useState<string[]>([]);
  const photographs = images.filter(
    (image) => image.image_url?.startsWith('/v1/') && !failed.includes(image.image_url),
  );
  const photograph = photographs[0];
  const generated = photograph?.provenance?.origin === 'ai_generated';
  const unavailableReason = failed.length
    ? 'The registered source photographs could not be loaded.'
    : images.find((image) => image.image_unavailable_reason)?.image_unavailable_reason ||
      (listedCount
        ? 'The source photographs are not available in this review.'
        : 'This claim does not list any source photographs.');

  return (
    <figure className="overview-photo-feature">
      {photograph?.image_url ? (
        <div className="overview-photo-stage">
          <img
            src={photograph.image_url}
            alt="Source claim photograph preview"
            onError={() => setFailed((current) => [...current, photograph.image_url!])}
          />
        </div>
      ) : (
        <div className="overview-photo-fallback" role="status">
          <span aria-hidden="true">▧</span>
          <h2>{listedCount || images.length ? 'Photos unavailable' : 'No photographs listed'}</h2>
          <p>{unavailableReason}</p>
          <p>Open the photograph details to inspect the recorded availability.</p>
        </div>
      )}
      <figcaption>
        <div>
          <strong>
            {photograph
              ? mode === 'demo'
                ? generated
                  ? 'AI-generated demo image'
                  : 'Prepared demo photo'
                : 'Source claim photograph'
              : 'Photographs'}
          </strong>
          <p>
            {status === 'AVAILABLE'
              ? 'Recorded vision results available.'
              : status === 'NO_EVIDENCE'
                ? 'No supported observations recorded.'
                : `Vision ${humanize(status).toLowerCase()}.`}
          </p>
        </div>
        <button type="button" className="overview-link" onClick={onOpen}>
          View photographs <span aria-hidden="true">↗</span>
        </button>
      </figcaption>
    </figure>
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
      <div className="overview-hero">
        <PhotoFeature
          key={view.claim.claim_id}
          images={view.vision.data?.images || []}
          listedCount={view.claim.images.length}
          status={view.vision.status}
          mode={view.mode}
          onOpen={() => onOpenDetails('vision')}
        />
        <div className="overview-review-summary">
          <section className="overview-next-step" aria-labelledby="overview-next-step-title">
            <p className="eyebrow">Suggested next step</p>
            <h2 id="overview-next-step-title">
              {recommendation
                ? humanize(recommendation.recommended_next_action)
                : 'No saved recommendation'}
            </h2>
            <p>
              {recommendation
                ? 'Previously recorded agent guidance.'
                : view.agent.reason ||
                  (view.agent.status === 'FAILED'
                    ? 'The agent run failed. Its recorded status is available in the details.'
                    : 'An agent recommendation has not been recorded for this review.')}
            </p>
            {recommendation ? (
              <button
                type="button"
                className="overview-link"
                onClick={() => onOpenDetails('agent')}
              >
                Read full recommendation <span aria-hidden="true">↗</span>
              </button>
            ) : (
              <StateBadge
                status={view.agent.status === 'AVAILABLE' ? 'NO_EVIDENCE' : view.agent.status}
              />
            )}
          </section>
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
            <button
              type="button"
              className="overview-link overview-signals-link"
              onClick={() => onOpenDetails('investigation')}
            >
              {signals.length > 3 ? `View all ${signals.length} signals` : 'View all signals'}{' '}
              <span aria-hidden="true">↗</span>
            </button>
          </section>
        </div>
      </div>
      <div className="overview-secondary">
        <section className="overview-evidence-summary" aria-label="Available case evidence">
          <div>
            <h2>Documents</h2>
            <p>
              {documents.length
                ? `${availableDocuments} available · ${missingDocuments} missing`
                : 'No documents listed'}
            </p>
          </div>
          <button
            type="button"
            className="overview-link"
            onClick={() => onOpenDetails('documents')}
          >
            Review documents <span aria-hidden="true">↗</span>
          </button>
        </section>
        {!!uncertainties.length && (
          <section
            className="overview-uncertainties"
            aria-labelledby="overview-uncertainties-title"
          >
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
    </div>
  );
}
