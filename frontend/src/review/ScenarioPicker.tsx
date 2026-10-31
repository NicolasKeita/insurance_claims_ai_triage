import type { DemoClaim } from '../types';
import { scenarioCopy } from './scenarios';

export function ScenarioPicker({
  claims,
  onSelect,
  selectedId,
  compact = false,
  disabled = false,
}: {
  claims: DemoClaim[];
  onSelect: (id: string) => void;
  selectedId?: string;
  compact?: boolean;
  disabled?: boolean;
}) {
  return (
    <div className={compact ? 'scenario-picker-compact' : 'scenario-grid'}>
      {claims.map((claim) => {
        const copy = scenarioCopy(claim);
        return (
          <button
            className={`scenario-card${selectedId === claim.claim_id ? ' selected' : ''}`}
            type="button"
            key={claim.claim_id}
            disabled={disabled}
            aria-label={`Open ${copy.title}`}
            aria-current={selectedId === claim.claim_id ? 'page' : undefined}
            onClick={() => onSelect(claim.claim_id)}
          >
            {!compact && <span className="scenario-kicker">{copy.focus}</span>}
            <strong>{copy.title}</strong>
            {!compact && <span className="scenario-description">{copy.description}</span>}
            {!compact && (
              <span className="scenario-card-cta">
                Open case <span aria-hidden="true">↗</span>
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
