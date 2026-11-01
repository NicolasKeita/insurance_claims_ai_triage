import { useI18n } from '../i18n';
import { useState } from 'react';
import { Badge, StateBadge, humanize } from '../presentation';
import type { ReviewView, VisionImage } from '../types';
import { recommendationConcerns } from '../review/recommendationConcerns';
import { reviewAssetUrl } from '../assets';

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
  const { t, translate } = useI18n();

  const [failed, setFailed] = useState<string[]>([]);
  const photographs = images
    .map((image) => ({ ...image, image_url: reviewAssetUrl(image.image_url, 'images') }))
    .filter((image) => image.image_url && !failed.includes(image.image_url));
  const photograph = photographs[0];
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
            alt={t('Source claim photograph preview')}
            onError={() => setFailed((current) => [...current, photograph.image_url!])}
          />
        </div>
      ) : (
        <div className="overview-photo-fallback" role="status">
          <span aria-hidden="true">▧</span>
          <h2>
            {t(listedCount || images.length ? 'Photos unavailable' : 'No photographs listed')}
          </h2>
          <p>{translate(unavailableReason)}</p>
          <p>{t('Open the photograph details to inspect the recorded availability.')}</p>
        </div>
      )}
      <figcaption>
        <div>
          <strong>
            {t(
              photograph
                ? mode === 'demo'
                  ? 'Demo image'
                  : 'Source claim photograph'
                : 'Photographs',
            )}
          </strong>
          <p>
            {status === 'AVAILABLE'
              ? t('Recorded vision results available.')
              : status === 'NO_EVIDENCE'
                ? t('No supported observations recorded.')
                : t('Vision {status}.', { status: humanize(status).toLowerCase() })}
          </p>
        </div>
        <button type="button" className="overview-link" onClick={onOpen}>
          {t('Check the photo evidence')}
          <span aria-hidden="true">↗</span>
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
  const { t, translate } = useI18n();

  const agent =
    view.agent.status === 'AVAILABLE' && view.agent.data?.status === 'COMPLETED'
      ? view.agent.data
      : null;
  const recommendation = agent?.recommendation;
  const investigation = view.investigation.status === 'AVAILABLE' ? view.investigation.data : null;
  const signals = investigation?.signals || [];
  const mainSignal = signals[0];
  const otherSignals = signals.slice(1);
  const documents = view.documents.data?.items || view.claim.documents;
  const availableDocuments = documents.filter((document) => document.available).length;
  const missingDocuments = documents.filter((document) => !document.available).length;
  const { points } = recommendationConcerns(recommendation?.uncertainties || []);

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
          <section className="overview-next-step" aria-labelledby="overview-result-title">
            <h2 id="overview-result-title">{t('Analysis results')}</h2>
            {mainSignal ? (
              <p className="overview-result-text">{translate(mainSignal.explanation)}</p>
            ) : recommendation?.case_summary ? (
              <p className="overview-result-text">{translate(recommendation.case_summary)}</p>
            ) : null}
            <p className="eyebrow overview-action-label">{t('Recommended next step')}</p>
            <h3 id="overview-next-step-title">
              {recommendation
                ? humanize(recommendation.recommended_next_action)
                : t('No saved recommendation')}
            </h3>
            {!recommendation && (
              <p>
                {translate(
                  view.agent.reason ||
                    (view.agent.status === 'FAILED'
                      ? 'The agent run failed. Its recorded status is available in the details.'
                      : 'An agent recommendation has not been recorded for this review.'),
                )}
              </p>
            )}
            {recommendation ? (
              <button
                type="button"
                className="primary-button overview-result-cta"
                onClick={() => onOpenDetails('agent')}
              >
                {t('Understand the recommendation')}
                <span aria-hidden="true">↗</span>
              </button>
            ) : (
              <StateBadge
                status={view.agent.status === 'AVAILABLE' ? 'NO_EVIDENCE' : view.agent.status}
              />
            )}
          </section>
          {(!investigation ||
            !signals.length ||
            otherSignals.length > 0 ||
            investigation.partial) && (
            <section className="overview-attention" aria-labelledby="overview-attention-title">
              <div className="overview-card-heading">
                <h2 id="overview-attention-title">{t('What needs attention')}</h2>
                {investigation ? (
                  <span className="overview-count">
                    {t(
                      signals.length === 1 ? '{count} recorded signal' : '{count} recorded signals',
                      { count: signals.length },
                    )}
                  </span>
                ) : (
                  <StateBadge status={view.investigation.status} />
                )}
              </div>
              {investigation ? (
                <>
                  {otherSignals.length ? (
                    <ul className="overview-signal-list">
                      {otherSignals.slice(0, 2).map((signal) => (
                        <li key={signal.code}>
                          <div>
                            <h3>
                              {signal.title ? translate(signal.title) : humanize(signal.code)}
                            </h3>
                            <Badge tone={signal.severity}>{humanize(signal.severity)}</Badge>
                          </div>
                          <p>{translate(signal.explanation)}</p>
                        </li>
                      ))}
                    </ul>
                  ) : !signals.length ? (
                    <p className="overview-empty">
                      {t('No review signals in this saved assessment.')}
                    </p>
                  ) : null}
                  {investigation.partial && (
                    <p className="overview-scope">
                      <strong>{t('Partial assessment.')}</strong>{' '}
                      {view.investigation.reason
                        ? translate(view.investigation.reason)
                        : investigation.unavailable_inputs?.length
                          ? t('Checks not run: {checks}.', {
                              checks: investigation.unavailable_inputs.map(humanize).join(', '),
                            })
                          : t('This saved assessment covers a limited set of checks.')}
                    </p>
                  )}
                </>
              ) : (
                <p className="overview-empty">
                  {translate(
                    view.investigation.reason ||
                      (view.investigation.status === 'FAILED'
                        ? 'The investigation assessment failed.'
                        : 'No investigation assessment is available. Missing results do not mean there are no review signals.'),
                  )}
                </p>
              )}
              <button
                type="button"
                className="overview-link overview-signals-link"
                onClick={() => onOpenDetails('investigation')}
              >
                {signals.length > 3
                  ? t('View all {count} signals', { count: signals.length })
                  : t('View all signals')}{' '}
                <span aria-hidden="true">↗</span>
              </button>
            </section>
          )}
        </div>
      </div>
      <div className="overview-secondary">
        <section className="overview-evidence-summary" aria-label={t('Available case evidence')}>
          <div>
            <h2>{t('Documents')}</h2>
            <p>
              {documents.length
                ? t('{available} available · {missing} missing', {
                    available: availableDocuments,
                    missing: missingDocuments,
                  })
                : t('No documents listed')}
            </p>
          </div>
          <button
            type="button"
            className="overview-link"
            onClick={() => onOpenDetails('documents')}
          >
            {t('Review documents')}
            <span aria-hidden="true">↗</span>
          </button>
        </section>
        {!!points.length && (
          <section
            className="overview-uncertainties"
            aria-labelledby="overview-uncertainties-title"
          >
            <div>
              <h2 id="overview-uncertainties-title">{t('Points to verify')}</h2>
              <p>
                {t(
                  points.length === 1
                    ? '{count} point to verify before making a decision.'
                    : '{count} points to verify before making a decision.',
                  { count: points.length },
                )}
              </p>
            </div>
            <button
              type="button"
              className="overview-link"
              onClick={() => onOpenDetails('agent-uncertainties')}
            >
              {t('View points to verify')}
              <span aria-hidden="true">↗</span>
            </button>
          </section>
        )}
      </div>
    </div>
  );
}
