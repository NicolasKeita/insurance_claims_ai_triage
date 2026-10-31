import { useI18n } from '../i18n';
import englishIllustration from '../assets/review-workflow-en-v2.png';
import frenchIllustration from '../assets/review-workflow-fr-v2.png';

/** The generated illustration explains the workflow; it is not claim evidence. */
export function DemoFlow() {
  const { t, locale } = useI18n();
  const workflowIllustration = locale === 'fr' ? frenchIllustration : englishIllustration;

  return (
    <figure className="demo-flow" aria-label={t('Illustrated review workflow')}>
      <div className="demo-flow-stage">
        <img
          src={workflowIllustration}
          width={2172}
          height={724}
          alt={t(
            'Claim file (PDFs and photos), Document AI and computer vision, LLM with policy evidence, then human review.',
          )}
        />
        <a href={workflowIllustration} target="_blank" rel="noreferrer">
          <span className="visually-hidden">{t('View full image')}</span>
          <span aria-hidden="true">↗</span>
        </a>
      </div>
    </figure>
  );
}
