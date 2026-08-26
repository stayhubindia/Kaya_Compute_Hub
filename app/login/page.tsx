'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { authClient } from '@/lib/api/authClient';

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const expired = searchParams.get('expired');

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (expired) {
      setErrorMessage('Your session has expired. Please log in again.');
    }
  }, [expired]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setIsLoading(true);

    try {
      await authClient.login({ email, password });
      router.push('/dashboard');
    } catch (err: any) {
      const msg = err.message || 'Authentication failed.';
      if (msg.includes('429') || msg.toLowerCase().includes('rate limit')) {
        setErrorMessage('Too many failed login attempts. Please wait a minute before trying again.');
      } else {
        setErrorMessage(msg);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{
      width: '100%',
      maxWidth: '440px',
      background: '#1e293b',
      border: '1px solid #334155',
      borderRadius: '12px',
      padding: '32px',
      boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)'
    }}>
      <div style={{ textAlign: 'center', marginBottom: '28px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', color: '#38bdf8', marginBottom: '8px' }}>
          Kaya Compute Hub
        </h1>
        <p style={{ fontSize: '14px', color: '#94a3b8' }}>
          Single-Admin Control Plane Sign In
        </p>
      </div>

      {errorMessage && (
        <div role="alert" style={{
          background: '#451a1a',
          border: '1px solid #991b1b',
          color: '#fca5a5',
          padding: '12px 16px',
          borderRadius: '8px',
          fontSize: '14px',
          marginBottom: '20px'
        }}>
          {errorMessage}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '18px' }}>
          <label htmlFor="email-input" style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '6px', color: '#cbd5e1' }}>
            Admin Email
          </label>
          <input
            id="email-input"
            type="text"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="admin@kaya.local"
            style={{
              width: '100%',
              padding: '10px 14px',
              background: '#0f172a',
              border: '1px solid #475569',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '14px',
              boxSizing: 'border-box'
            }}
          />
        </div>

        <div style={{ marginBottom: '18px' }}>
          <label htmlFor="password-input" style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '6px', color: '#cbd5e1' }}>
            Admin Password
          </label>
          <input
            id="password-input"
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••••••"
            style={{
              width: '100%',
              padding: '10px 14px',
              background: '#0f172a',
              border: '1px solid #475569',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '14px',
              boxSizing: 'border-box'
            }}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          style={{
            width: '100%',
            padding: '12px',
            background: '#0284c7',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            fontSize: '15px',
            fontWeight: '600',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            opacity: isLoading ? 0.7 : 1,
            marginTop: '8px'
          }}
        >
          {isLoading ? 'Authenticating...' : 'Sign In to Control Plane'}
        </button>
      </form>
    </div>
  );
}

export default function LoginPage() {
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      color: '#f8fafc',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      padding: '20px'
    }}>
      <Suspense fallback={<div style={{ color: '#94a3b8' }}>Loading Login Form...</div>}>
        <LoginForm />
      </Suspense>
    </div>
  );
}
