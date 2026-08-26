import React from 'react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({ title = 'An error occurred', message, onRetry }) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center rounded-xl bg-rose-950/20 border border-rose-800/40 text-rose-300">
      <div className="w-10 h-10 rounded-full bg-rose-900/50 flex items-center justify-center mb-3 text-rose-400">
        ⚠️
      </div>
      <h3 className="text-sm font-semibold text-rose-200 mb-1">{title}</h3>
      <p className="text-xs text-rose-300/80 max-w-md mb-4">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-rose-800 hover:bg-rose-700 text-white transition-colors"
        >
          Try Again
        </button>
      )}
    </div>
  );
};
