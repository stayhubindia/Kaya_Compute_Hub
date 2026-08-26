'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { authClient } from '@/lib/api/authClient';

export default function RootPage() {
  const router = useRouter();

  useEffect(() => {
    async function checkAuth() {
      try {
        await authClient.getCurrentUser();
        router.push('/dashboard');
      } catch {
        router.push('/login');
      }
    }
    checkAuth();
  }, [router]);

  return (
    <div style={{
      minHeight: '100vh',
      background: '#090d16',
      color: '#38bdf8',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'system-ui'
    }}>
      Redirecting to Kaya Compute Hub...
    </div>
  );
}
