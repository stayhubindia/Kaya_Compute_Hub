import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Kaya Compute Hub | VM Control Plane',
  description: 'Secure, private, web-accessible VM control plane for long-running compute workloads.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
