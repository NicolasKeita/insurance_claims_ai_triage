import { useI18n } from '../i18n';
import { humanize } from '../presentation';
import type { ReviewView } from '../types';

export function DamageComparison({ view }: { view: ReviewView }) {
  const { t } = useI18n();
  const additionalDamage =
    view.visual_consistency.status === 'AVAILABLE'
      ? view.visual_consistency.data?.additional_visual_damage || []
      : [];

  return (
    <dl className="damage-comparison" aria-label={t('Declared and additional damage')}>
      <div>
        <dt>{t('Declared damage')}</dt>
        <dd>
          {view.claim.declared_damage.length
            ? view.claim.declared_damage.map(humanize).join(', ')
            : t('Not recorded')}
        </dd>
      </div>
      {!!additionalDamage.length && (
        <div>
          <dt>{t('Additional damage observed in photographs (to verify)')}</dt>
          <dd>{additionalDamage.map(humanize).join(', ')}</dd>
        </div>
      )}
    </dl>
  );
}
