import { useState } from 'react';
import { Badge, Metric, Section, Tags, anchor, humanize, percent } from '../presentation';
import type { Polygon, ReviewView, VisionImage } from '../types';

function polygonPath(polygon: Polygon): string {
  const ring = (points: Polygon['points']) =>
    `M ${points.map(({ x, y }) => `${x * 100} ${y * 100}`).join(' L ')} Z`;
  return [
    ring(polygon.points),
    ...(polygon.holes || []).map(ring),
    ...(polygon.components || []).map(polygonPath),
  ].join(' ');
}

function EvidenceImage({ image }: { image: VisionImage }) {
  const [failed, setFailed] = useState(false);
  const filename = image.filename || image.damage_assessment?.filename || 'Claim photograph';
  const safeUrl = image.image_url?.startsWith('/v1/') ? image.image_url : null;
  if (!safeUrl || failed)
    return (
      <div className="photo-unavailable">
        <span aria-hidden="true">▧</span>
        <strong>Image unavailable</strong>
        <p>
          {failed
            ? 'The image file could not be loaded.'
            : image.image_unavailable_reason ||
              'The source photograph is not available through the configured claim assets.'}
        </p>
        <small>{filename}</small>
      </div>
    );
  return (
    <figure className="evidence-image">
      <div className="photo-stage">
        <img src={safeUrl} alt={`Claim photograph: ${filename}`} onError={() => setFailed(true)} />
        {!!image.damage_assessment?.detections.length && (
          <svg
            className="damage-overlay"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            role="img"
            aria-label="Recorded damage masks and bounding boxes"
          >
            {image.damage_assessment.detections.map((detection, index) => {
              const box = detection.bounding_box;
              return (
                <g key={index}>
                  {detection.polygon && (
                    <path d={polygonPath(detection.polygon)} fillRule="evenodd" />
                  )}
                  {box && (
                    <rect
                      x={box.x_min * 100}
                      y={box.y_min * 100}
                      width={(box.x_max - box.x_min) * 100}
                      height={(box.y_max - box.y_min) * 100}
                    />
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <figcaption>
        <span>{filename}</span>
        <span>
          {image.damage_assessment?.detections.length
            ? 'Recorded evidence overlay'
            : 'Source photograph · no recorded overlay'}
        </span>
      </figcaption>
    </figure>
  );
}

export function VisualEvidence({ view }: { view: ReviewView }) {
  return (
    <Section
      id="vision"
      title="Visual Evidence"
      eyebrow="05 / Observed image evidence"
      section={view.vision}
      help="Photographs and recorded detector observations. Opening this review does not run a vision model."
    >
      {(vision) => (
        <div className="visual-images">
          {vision.images.map((image, imageIndex) => {
            const filename =
              image.filename || image.damage_assessment?.filename || `Image ${imageIndex + 1}`;
            return (
              <article className="visual-image" key={filename}>
                <EvidenceImage image={image} />
                <div className="visual-observations">
                  {!image.evidence.length && (
                    <p className="muted">No stored observations for this image.</p>
                  )}
                  {image.evidence.map((observation, index) => (
                    <article
                      className="observation"
                      id={anchor(`vision:${filename}:obs_${index + 1}`)}
                      key={observation.damage_detection_index}
                      tabIndex={-1}
                    >
                      <div className="observation-heading">
                        <span className="observation-number">
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <h3>{humanize(observation.visual_damage_type)}</h3>
                        <Badge tone={observation.association_status}>
                          {humanize(observation.association_status)}
                        </Badge>
                      </div>
                      <dl className="metrics observation-metrics">
                        <Metric
                          label="Vehicle part"
                          value={
                            observation.vehicle_part
                              ? humanize(observation.vehicle_part)
                              : 'Not associated'
                          }
                        />
                        <Metric
                          label="Damage confidence"
                          value={percent(observation.damage_confidence)}
                        />
                        <Metric
                          label="Part confidence"
                          value={percent(observation.part_confidence)}
                        />
                        <Metric
                          label="Containment"
                          value={percent(observation.containment_ratio)}
                        />
                      </dl>
                      <code className="observation-id">
                        vision:{filename}:obs_{index + 1}
                      </code>
                    </article>
                  ))}
                </div>
              </article>
            );
          })}
          {!vision.images.length && (
            <p className="muted">No source images are listed in this snapshot.</p>
          )}
        </div>
      )}
    </Section>
  );
}

export function VisualConsistency({ view }: { view: ReviewView }) {
  return (
    <Section
      id="visual_consistency"
      title="Visual Consistency"
      section={view.visual_consistency}
      help="Not visually confirmed does not mean the declared damage is absent."
    >
      {(consistency) => (
        <div className="visual-consistency-list">
          <div>
            <h3>Visually supported declared damage</h3>
            <Tags items={consistency.visually_supported_declared_damage} />
          </div>
          <div>
            <h3>Visually supported quote damage</h3>
            <Tags items={consistency.visually_supported_quote_damage} />
          </div>
          <div>
            <h3>Additional visual damage</h3>
            <Tags items={consistency.additional_visual_damage} />
          </div>
          <div>
            <h3>Declared damage not visually confirmed</h3>
            <Tags
              items={consistency.declared_damage_not_visually_confirmed}
              empty="None recorded in this assessment"
            />
          </div>
          <div>
            <h3>Unmapped visual damage</h3>
            <Tags
              items={consistency.visual_damage_unmapped.map(
                (item) => `${humanize(item.visual_damage_type)} · ${item.filename}`,
              )}
            />
          </div>
        </div>
      )}
    </Section>
  );
}
