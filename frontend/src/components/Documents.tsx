import { Badge, FactList, Section, Tags, humanize } from '../presentation';
import { useI18n } from '../i18n';
import type { ReviewView } from '../types';
import { reviewAssetUrl } from '../assets';

export function Documents({ view }: { view: ReviewView }) {
  const { t, translate } = useI18n();
  return (
    <Section
      id="documents"
      title={t('Documents')}
      eyebrow={t('01 / Source material')}
      section={view.documents}
    >
      {(documents) => (
        <div className="document-list">
          {documents.items.map((document, index) => {
            const extraction = document.extracted_data || documents.extractions?.[document.type];
            const documentUrl =
              document.available && document.filename?.toLowerCase().endsWith('.pdf')
                ? reviewAssetUrl(document.document_url, 'documents')
                : null;
            return (
              <article className="document" key={`${document.type}-${index}`}>
                <div className="document-row">
                  <span className="document-icon" aria-hidden="true">
                    ▤
                  </span>
                  <div className="document-name">
                    <h3>{humanize(document.type)}</h3>
                    <p>{document.filename || t('No file supplied')}</p>
                  </div>
                  <Badge tone={document.available ? 'available' : 'missing'}>
                    {document.available ? t('Available') : t('Missing')}
                  </Badge>
                </div>
                <div className="document-meta">
                  {documentUrl ? (
                    <a
                      className="overview-link"
                      href={documentUrl}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={t('Open PDF: {filename}', { filename: document.filename! })}
                    >
                      {t('Open PDF')}
                      <span aria-hidden="true">↗</span>
                    </a>
                  ) : document.document_unavailable_reason ? (
                    <span>
                      {t('Original PDF unavailable.')}{' '}
                      {translate(document.document_unavailable_reason)}
                    </span>
                  ) : null}
                  <span>
                    {t('Extraction')}{' '}
                    <strong>
                      {document.extraction_method
                        ? humanize(document.extraction_method)
                        : t('Not recorded')}
                    </strong>
                  </span>
                  <span>
                    {t('Source')}{' '}
                    <strong>
                      {document.source ? translate(document.source) : t('Claim document manifest')}
                    </strong>
                  </span>
                </div>
                {extraction && (
                  <details className="extraction">
                    <summary>{t('Extracted fields')}</summary>
                    <FactList facts={extraction} />
                  </details>
                )}
              </article>
            );
          })}
          {!documents.items.length && (
            <p className="muted">{t('No documents are listed in this claim.')}</p>
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
  const { t } = useI18n();
  return (
    <Section
      id="consistency"
      title={t('Document Consistency')}
      section={view.consistency}
      help={t(
        'Factual comparisons across the submitted claim form, accident report and repair quote.',
      )}
    >
      {(report) => (
        <>
          <div className="comparison-grid">
            {comparisons.map(([field, label]) => (
              <div className="comparison" key={field}>
                <span>{t(label)}</span>
                <Badge tone={report[field] ? 'available' : 'elevated'}>
                  {report[field] === true
                    ? t('Matches')
                    : report[field] === false
                      ? t('Mismatch')
                      : t('Not assessed')}
                </Badge>
              </div>
            ))}
          </div>
          <div className="fact-block">
            <h3>{t('Additional quote damage')}</h3>
            <Tags
              items={report.additional_quote_damage}
              empty={t('No additional damage in the recorded comparison')}
            />
          </div>
          <div className="fact-block">
            <h3>{t('Unmapped quote items')}</h3>
            <Tags items={report.unmapped_quote_items} />
          </div>
        </>
      )}
    </Section>
  );
}
