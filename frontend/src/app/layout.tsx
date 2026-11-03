import type { Metadata, Viewport } from 'next';
import type { ReactNode } from 'react';
import { themeBootstrap } from '../theme-bootstrap';
import './globals.css';

export const metadata: Metadata = {
  title: 'Claim Review · Insurance AI',
  description: 'Explore prepared insurance claims, their findings and supporting evidence.',
};

export const viewport: Viewport = { themeColor: '#10181d' };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="fr" data-theme="dark" suppressHydrationWarning>
      <head>
        <script id="theme-bootstrap" dangerouslySetInnerHTML={{ __html: themeBootstrap }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
