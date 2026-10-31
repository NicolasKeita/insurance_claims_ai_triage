import { useI18n } from '../i18n';

export function LanguageToggle() {
  const { locale, setLocale, t } = useI18n();
  return (
    <div className="language-toggle" role="group" aria-label={t('Language')}>
      <button
        type="button"
        lang="fr"
        aria-label="Français"
        aria-pressed={locale === 'fr'}
        onClick={() => setLocale('fr')}
      >
        FR
      </button>
      <button
        type="button"
        lang="en"
        aria-label="English"
        aria-pressed={locale === 'en'}
        onClick={() => setLocale('en')}
      >
        EN
      </button>
    </div>
  );
}
