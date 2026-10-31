import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Documents } from '../components/Documents';
import { setLocale } from '../i18n';
import { partialReview } from './fixtures';

describe('Copied document sources', () => {
  it.each([
    { locale: 'en' as const, label: 'Open PDF: claim_form.pdf', text: 'Open PDF' },
    { locale: 'fr' as const, label: 'Ouvrir le PDF : claim_form.pdf', text: 'Ouvrir le PDF' },
  ])(
    'opens the registered PDF in $locale without changing its review',
    ({ locale, label, text }) => {
      setLocale(locale, false);
      const view = partialReview();
      const documentUrl = '/v1/claims/DEMO-COHERENT/review/documents/claim_form.pdf';
      view.documents.data!.items[0].document_url = documentUrl;
      const original = JSON.stringify(view);
      render(<Documents view={view} />);
      const link = screen.getByRole('link', { name: label });
      expect(link).toHaveTextContent(text);
      expect(link).toHaveAttribute('href', documentUrl);
      expect(link).toHaveAttribute('target', '_blank');
      expect(link).toHaveAttribute('rel', 'noreferrer');
      expect(JSON.stringify(view)).toBe(original);
    },
  );

  it('reports a missing copied original separately from the recorded document inventory', () => {
    const view = partialReview();
    view.documents.data!.items[0].document_url = null;
    view.documents.data!.items[0].document_unavailable_reason =
      'The original document was not retained in this archive.';
    const original = JSON.stringify(view);
    render(<Documents view={view} />);
    expect(within(screen.getByRole('article')).getByText('Available')).toBeVisible();
    expect(
      screen.getByText(
        'Original PDF unavailable. The original document was not retained in this archive.',
      ),
    ).toBeVisible();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(JSON.stringify(view)).toBe(original);
  });

  it.each([
    'https://example.com/claim_form.pdf',
    '//example.com/claim_form.pdf',
    'javascript:alert(1)',
    'file:///C:/claims/claim_form.pdf',
    'documents/claim_form.pdf',
  ])('does not create a source link for an unregistered URL: %s', (url) => {
    const view = partialReview();
    view.documents.data!.items[0].document_url = url;
    render(<Documents view={view} />);
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('keeps an inventory-only document without inventing a source URL or absence reason', () => {
    render(<Documents view={partialReview()} />);
    expect(screen.getByText('claim_form.pdf')).toBeVisible();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.queryByText('Original PDF unavailable.')).not.toBeInTheDocument();
  });

  it('localizes the missing archive PDF message in French', () => {
    setLocale('fr', false);
    const view = partialReview();
    view.documents.data!.items[0].document_unavailable_reason =
      'The original PDF is not included in this archive.';
    render(<Documents view={view} />);
    expect(
      screen.getByText(
        'PDF original indisponible. Le PDF original n’est pas inclus dans cette archive.',
      ),
    ).toBeVisible();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it.each([
    { available: false, filename: 'claim_form.pdf' },
    { available: true, filename: 'claim_form.html' },
    { available: true, filename: null },
  ])('does not link an unavailable or non-PDF source: $filename / $available', (record) => {
    const view = partialReview();
    Object.assign(view.documents.data!.items[0], record, {
      document_url: '/v1/claims/DEMO-COHERENT/review/documents/claim_form.pdf',
    });
    render(<Documents view={view} />);
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });
});
