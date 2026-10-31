import { useI18n } from '../i18n';
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
  const { t } = useI18n();

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
            aria-label={t('Open {title}', { title: t(copy.title) })}
            aria-current={selectedId === claim.claim_id ? 'page' : undefined}
            onClick={() => onSelect(claim.claim_id)}
          >
            {!compact && <span className="scenario-kicker">{t(copy.focus)}</span>}
            <strong>{t(copy.title)}</strong>
            {!compact && <span className="scenario-description">{t(copy.description)}</span>}
            {!compact && (
              <span className="scenario-card-cta">
                {t('Open case')}
                <span aria-hidden="true">↗</span>
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
