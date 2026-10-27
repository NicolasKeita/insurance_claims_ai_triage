import { Badge, FactList, Section, Tags, humanize } from '../presentation';
import type { ReviewView } from '../types';

export function Documents({ view }: { view: ReviewView }) {
  return (
    <Section
      id="documents"
      title="Documents"
      eyebrow="01 / Source material"
      section={view.documents}
    >
      {(documents) => (
        <div className="document-list">
          {documents.items.map((document, index) => {
            const extraction = document.extracted_data || documents.extractions?.[document.type];
            return (
              <article className="document" key={`${document.type}-${index}`}>
                <div className="document-row">
                  <span className="document-icon" aria-hidden="true">
                    ▤
                  </span>
                  <div className="document-name">
                    <h3>{humanize(document.type)}</h3>
                    <p>{document.filename || 'No file supplied'}</p>
                  </div>
                  <Badge tone={document.available ? 'available' : 'missing'}>
                    {document.available ? 'Available' : 'Missing'}
                  </Badge>
                </div>
                <div className="document-meta">
                  <span>
                    Extraction{' '}
                    <strong>
                      {document.extraction_method
                        ? humanize(document.extraction_method)
                        : 'Not recorded'}
                    </strong>
                  </span>
                  <span>
                    Source <strong>{document.source || 'Claim document manifest'}</strong>
                  </span>
                </div>
                {extraction && (
                  <details className="extraction">
                    <summary>Extracted fields</summary>
                    <FactList facts={extraction} />
                  </details>
                )}
              </article>
            );
          })}
          {!documents.items.length && (
            <p className="muted">No documents are listed in this claim.</p>
          )}
        </div>
      )}
    </Section>
  );
}

const comparisons = [
  ['claim_id_match', 'Claim ID'],
  ['incident_date_match', 'Incident date'],
  ['location_match', 'Location'],
  ['vehicle_match', 'Vehicle'],
  ['collision_type_match', 'Collision type'],
  ['injuries_match', 'Injuries declaration'],
  ['declared_damage_match', 'Declared damage'],
] as const;

export function DocumentConsistency({ view }: { view: ReviewView }) {
  return (
    <Section
      id="consistency"
      title="Document Consistency"
      section={view.consistency}
      help="Factual comparisons across the submitted claim form, accident report and repair quote."
    >
      {(report) => (
        <>
          <div className="comparison-grid">
            {comparisons.map(([field, label]) => (
              <div className="comparison" key={field}>
                <span>{label}</span>
                <Badge tone={report[field] ? 'available' : 'elevated'}>
                  {report[field] === true
                    ? 'Matches'
                    : report[field] === false
                      ? 'Mismatch'
                      : 'Not assessed'}
                </Badge>
              </div>
            ))}
          </div>
          <div className="fact-block">
            <h3>Additional quote damage</h3>
            <Tags
              items={report.additional_quote_damage}
              empty="No additional damage in the recorded comparison"
            />
          </div>
          <div className="fact-block">
            <h3>Unmapped quote items</h3>
            <Tags items={report.unmapped_quote_items} />
          </div>
        </>
      )}
    </Section>
  );
}
