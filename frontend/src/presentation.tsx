import type { ReactNode } from 'react';
import { currentLocale, translateText, useI18n } from './i18n';
import type { Facts, ReviewSection, ReviewStatus } from './types';

const labels: Record<ReviewStatus, string> = {
  AVAILABLE: 'Available',
  UNAVAILABLE: 'Unavailable',
  NOT_EXECUTED: 'Not executed',
  FAILED: 'Failed',
  NO_EVIDENCE: 'No evidence',
  INSUFFICIENT_EVIDENCE: 'Insufficient evidence',
};

export function humanize(value: string): string {
  const exact = translateText(value);
  if (exact !== value) return exact;
  return translateText(
    value
      .replaceAll('_', ' ')
      .toLowerCase()
      .replace(/^./, (letter) => letter.toUpperCase()),
  );
}

export function decimal(value: number | null | undefined, places = 3): string {
  return value == null
    ? translateText('Not available')
    : value.toLocaleString(currentLocale() === 'fr' ? 'fr-FR' : 'en-GB', {
        maximumFractionDigits: places,
      });
}

export function percent(value: number | null | undefined): string {
  return value == null
    ? translateText('Not available')
    : new Intl.NumberFormat(currentLocale() === 'fr' ? 'fr-FR' : 'en-GB', {
        style: 'percent',
        maximumFractionDigits: 1,
      }).format(value);
}

/** Format explicitly labelled confidence in prose without changing the saved narrative. */
export function narrative(value: string): string {
  return translateText(value).replace(
    /\b((?:confidence of|confiance de)\s+)(0[.,]\d+|1[.,]0+)(?!\d|[.,]\d)/gi,
    (_match, label: string, score: string) => `${label}${percent(Number(score.replace(',', '.')))}`,
  );
}

/** Linked evidence carries verified references; keep technical citations out of the prose. */
export function rationaleText(value: string, evidenceIds: string[]): string {
  const retainedIds = new Set(evidenceIds);
  return narrative(value)
    .replace(
      /\s*(?:The evidence_id is|La référence evidence_id est)\s+(\S+?)\.?\s*$/i,
      (match, id: string) => (retainedIds.has(id) ? '' : match),
    )
    .replace(/\s*\(evidence_ids?\s*:\s*([^)]+)\)/gi, (match, references: string) =>
      references.split(/\s*,\s*/).every((id) => retainedIds.has(id.trim())) ? '' : match,
    );
}

export function money(value: number | string | null | undefined, currency?: string | null): string {
  if (value == null || !currency || !Number.isFinite(Number(value)))
    return translateText('Not available');
  return new Intl.NumberFormat(currentLocale() === 'fr' ? 'fr-FR' : 'en-GB', {
    style: 'currency',
    currency,
  }).format(Number(value));
}

export function date(value: string | null | undefined, time = false): string {
  if (!value) return translateText('Not recorded');
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat(currentLocale() === 'fr' ? 'fr-FR' : 'en-GB', {
        dateStyle: 'medium',
        ...(/^\d{4}-\d{2}-\d{2}$/.test(value) ? { timeZone: 'UTC' } : {}),
        ...(time ? { timeStyle: 'short' as const } : {}),
      }).format(parsed);
}

export function Badge({ children, tone = '' }: { children: ReactNode; tone?: string }) {
  const { translate } = useI18n();
  return (
    <span className={`badge ${tone.toLowerCase()}`}>
      {typeof children === 'string' ? translate(children) : children}
    </span>
  );
}

export function StateBadge({ status }: { status: ReviewStatus }) {
  return <Badge tone={status}>{labels[status]}</Badge>;
}

export function Section<T>({
  id,
  title,
  eyebrow,
  section,
  children,
  className = '',
  help,
}: {
  id: string;
  title: string;
  eyebrow?: string;
  section: ReviewSection<T>;
  children: (data: T) => ReactNode;
  className?: string;
  help?: string;
}) {
  const { t, translate } = useI18n();
  const displayData = section.data != null && section.status !== 'FAILED';
  return (
    <section id={id} className={`panel ${className}`} aria-labelledby={`${id}-title`} tabIndex={-1}>
      <div className="panel-heading">
        <div>
          {eyebrow && <p className="eyebrow">{t(eyebrow)}</p>}
          <h2 id={`${id}-title`}>{t(title)}</h2>
        </div>
        <StateBadge status={section.status} />
      </div>
      {help && <p className="section-help">{t(help)}</p>}
      {section.status === 'AVAILABLE' && section.reason && (
        <p className="scope-note">{translate(section.reason)}</p>
      )}
      {section.status !== 'AVAILABLE' && (
        <div className={`empty-state ${section.status.toLowerCase()}`} role="status">
          <span className="state-icon" aria-hidden="true">
            {section.status === 'FAILED' ? '!' : '—'}
          </span>
          <div>
            <strong>{t(labels[section.status])}</strong>
            <p>{translate(section.reason || 'No result has been recorded for this component.')}</p>
          </div>
        </div>
      )}
      {displayData && children(section.data as T)}
      {(section.source || section.updated_at) && (
        <p className="provenance-line">
          {section.source && <span>{t('Source: {source}', { source: section.source })}</span>}
          {section.updated_at && (
            <span>{t('Recorded {date}', { date: date(section.updated_at, true) })}</span>
          )}
        </p>
      )}
    </section>
  );
}

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  const { t, translate } = useI18n();
  return (
    <div className="metric">
      <dt>{t(label)}</dt>
      <dd>{typeof value === 'string' ? translate(value) : value}</dd>
    </div>
  );
}

export function Tags({ items, empty = 'None recorded' }: { items?: string[]; empty?: string }) {
  const { t } = useI18n();
  return !items?.length ? (
    <span className="muted">{t(empty)}</span>
  ) : (
    <ul className="tag-list">
      {items.map((item, index) => (
        <li key={`${item}-${index}`}>{humanize(item)}</li>
      ))}
    </ul>
  );
}

function Value({ value, depth }: { value: unknown; depth: number }) {
  const { t, translate } = useI18n();
  if (value == null) return <span className="muted">{t('Not recorded')}</span>;
  if (typeof value === 'boolean') return <span>{t(value ? 'Yes' : 'No')}</span>;
  if (typeof value === 'number') return <span>{decimal(value, 4)}</span>;
  if (typeof value === 'string') return <span className="fact-value">{translate(value)}</span>;
  if (depth > 5) return <span className="muted">{t('See source record for further detail')}</span>;
  if (Array.isArray(value)) {
    return value.length ? (
      <ul className="fact-list">
        {value.map((item, i) => (
          <li key={i}>
            <Value value={item} depth={depth + 1} />
          </li>
        ))}
      </ul>
    ) : (
      <span className="muted">{t('None recorded')}</span>
    );
  }
  if (typeof value === 'object') return <FactList facts={value as Facts} depth={depth + 1} />;
  return <span className="muted">{t('Not recorded')}</span>;
}

/** Structured display only: no scoring, inference or domain decisions occur here. */
export function FactList({ facts, depth = 0 }: { facts: Facts; depth?: number }) {
  useI18n();
  return (
    <dl className="facts">
      {Object.entries(facts).map(([key, value]) => (
        <div key={key}>
          <dt>{humanize(key)}</dt>
          <dd>
            <Value value={value} depth={depth} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function anchor(id: string): string {
  return `evidence-${encodeURIComponent(id)}`;
}

export function EvidenceLinks({ ids, onOpen }: { ids: string[]; onOpen: (id: string) => void }) {
  const { t } = useI18n();
  return (
    <span className="evidence-links">
      {ids.map((id) => (
        <button
          type="button"
          className="evidence-link"
          key={id}
          onClick={() => onOpen(id)}
          aria-label={t('Open evidence {id}', { id })}
        >
          <span aria-hidden="true">↗</span> {id}
        </button>
      ))}
    </span>
  );
}
