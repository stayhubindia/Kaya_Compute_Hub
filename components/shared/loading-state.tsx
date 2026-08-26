import React from 'react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ message = 'Loading state...' }) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center rounded-xl bg-slate-900/50 border border-slate-800">
      <div className="w-8 h-8 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mb-3" />
      <p className="text-sm font-medium text-slate-400">{message}</p>
    </div>
  );
};
