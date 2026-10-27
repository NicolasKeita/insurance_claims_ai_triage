import type { ReactNode } from 'react';
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
  return value
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/^./, (letter) => letter.toUpperCase());
}

export function decimal(value: number | null | undefined, places = 3): string {
  return value == null
    ? 'Not available'
    : value.toLocaleString('en-GB', { maximumFractionDigits: places });
}

export function percent(value: number | null | undefined): string {
  return value == null ? 'Not available' : `${decimal(value * 100, 1)}%`;
}

export function money(value: number | string | null | undefined, currency?: string | null): string {
  if (value == null || !currency) return 'Not available';
  return new Intl.NumberFormat('en-GB', { style: 'currency', currency }).format(Number(value));
}

export function date(value: string | null | undefined, time = false): string {
  if (!value) return 'Not recorded';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat('en-GB', {
        dateStyle: 'medium',
        ...(time ? { timeStyle: 'short' as const } : {}),
      }).format(parsed);
}

export function Badge({ children, tone = '' }: { children: ReactNode; tone?: string }) {
  return <span className={`badge ${tone.toLowerCase()}`}>{children}</span>;
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
  const displayData = section.data != null && section.status !== 'FAILED';
  return (
    <section id={id} className={`panel ${className}`} aria-labelledby={`${id}-title`} tabIndex={-1}>
      <div className="panel-heading">
        <div>
          {eyebrow && <p className="eyebrow">{eyebrow}</p>}
          <h2 id={`${id}-title`}>{title}</h2>
        </div>
        <StateBadge status={section.status} />
      </div>
      {help && <p className="section-help">{help}</p>}
      {section.status === 'AVAILABLE' && section.reason && (
        <p className="scope-note">{section.reason}</p>
      )}
      {section.status !== 'AVAILABLE' && (
        <div className={`empty-state ${section.status.toLowerCase()}`} role="status">
          <span className="state-icon" aria-hidden="true">
            {section.status === 'FAILED' ? '!' : '—'}
          </span>
          <div>
            <strong>{labels[section.status]}</strong>
            <p>{section.reason || 'No result has been recorded for this component.'}</p>
          </div>
        </div>
      )}
      {displayData && children(section.data as T)}
      {(section.source || section.updated_at) && (
        <p className="provenance-line">
          {section.source && <span>Source: {section.source}</span>}
          {section.updated_at && <span>Recorded {date(section.updated_at, true)}</span>}
        </p>
      )}
    </section>
  );
}

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export function Tags({ items, empty = 'None recorded' }: { items?: string[]; empty?: string }) {
  return !items?.length ? (
    <span className="muted">{empty}</span>
  ) : (
    <ul className="tag-list">
      {items.map((item, index) => (
        <li key={`${item}-${index}`}>{humanize(item)}</li>
      ))}
    </ul>
  );
}

function Value({ value, depth }: { value: unknown; depth: number }) {
  if (value == null) return <span className="muted">Not recorded</span>;
  if (typeof value === 'boolean') return <span>{value ? 'Yes' : 'No'}</span>;
  if (typeof value === 'number') return <span>{decimal(value, 4)}</span>;
  if (typeof value === 'string') return <span className="fact-value">{value}</span>;
  if (depth > 5) return <span className="muted">See source record for further detail</span>;
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
      <span className="muted">None recorded</span>
    );
  }
  if (typeof value === 'object') return <FactList facts={value as Facts} depth={depth + 1} />;
  return <span className="muted">Not recorded</span>;
}

/** Structured display only: no scoring, inference or domain decisions occur here. */
export function FactList({ facts, depth = 0 }: { facts: Facts; depth?: number }) {
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
  return (
    <span className="evidence-links">
      {ids.map((id) => (
        <button type="button" className="evidence-link" key={id} onClick={() => onOpen(id)}>
          <span aria-hidden="true">↗</span> {id}
        </button>
      ))}
    </span>
  );
}
