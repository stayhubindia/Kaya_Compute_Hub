import React from 'react';

interface EmptyStateProps {
  title: string;
  description: string;
  actionText?: string;
  onAction?: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ title, description, actionText, onAction }) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center rounded-xl bg-slate-900/40 border border-slate-800/80">
      <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center mb-4 text-slate-500">
        📦
      </div>
      <h3 className="text-base font-semibold text-slate-200 mb-1">{title}</h3>
      <p className="text-sm text-slate-400 max-w-sm mb-4">{description}</p>
      {actionText && onAction && (
        <button
          onClick={onAction}
          className="px-4 py-2 text-xs font-medium rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white transition-colors"
        >
          {actionText}
        </button>
      )}
    </div>
  );
};
