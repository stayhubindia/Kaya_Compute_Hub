'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { User, authClient } from '@/lib/api/authClient';

interface Props {
  user: User | null;
}

export default function DashboardNavbar({ user }: Props) {
  const pathname = usePathname();
  const router = useRouter();

  const handleLogout = async () => {
    try {
      await authClient.logout();
    } catch {
      // Ignore network error on logout
    } finally {
      router.push('/login');
    }
  };

  const navItems = [
    { href: '/dashboard', label: 'Overview' },
    { href: '/dashboard/console', label: 'Console & Runner' },
    { href: '/dashboard/factory', label: 'Dataset Factory' },
    { href: '/dashboard/jobs', label: 'Jobs' },
    { href: '/dashboard/workers', label: 'Workers' },
    { href: '/dashboard/datasets', label: 'Datasets' },
    { href: '/dashboard/artifacts', label: 'Artifacts' },
    { href: '/dashboard/integrations', label: 'Integrations' },
    { href: '/dashboard/settings/connections', label: 'Connections' },
  ];

  return (
    <header style={{
      background: '#0f172a',
      borderBottom: '1px solid #1e293b',
      padding: '12px 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      color: '#f8fafc',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
        <Link href="/dashboard" style={{ textDecoration: 'none', fontSize: '18px', fontWeight: '700', color: '#38bdf8' }}>
          Kaya Compute Hub
        </Link>
        <nav style={{ display: 'flex', gap: '8px' }}>
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  textDecoration: 'none',
                  color: isActive ? '#fff' : '#94a3b8',
                  background: isActive ? '#1e293b' : 'transparent',
                  border: isActive ? '1px solid #334155' : '1px solid transparent'
                }}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {user && (
          <div style={{ fontSize: '13px', color: '#cbd5e1', textAlign: 'right' }}>
            <div>{user.email}</div>
            <div style={{ fontSize: '11px', color: '#38bdf8', textTransform: 'uppercase', fontWeight: '600' }}>
              Administrator
            </div>
          </div>
        )}
        <button
          onClick={handleLogout}
          style={{
            padding: '6px 12px',
            background: '#334155',
            color: '#f8fafc',
            border: 'none',
            borderRadius: '6px',
            fontSize: '13px',
            cursor: 'pointer'
          }}
        >
          Sign Out
        </button>
      </div>
    </header>
  );
}
