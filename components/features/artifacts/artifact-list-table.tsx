'use client';

import React from 'react';
import { ArtifactItem } from '../../../lib/schemas/dashboard-schemas';

interface ArtifactListTableProps {
  artifacts: ArtifactItem[];
}

export const ArtifactListTable: React.FC<ArtifactListTableProps> = ({ artifacts }) => {
  const handleDownload = (artifact: ArtifactItem) => {
    // Initiate secure download request
    window.open(`/api/v1/artifacts/${artifact.id}/download/`, '_blank');
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  if (!artifacts || artifacts.length === 0) {
    return (
      <div className="p-8 text-center bg-slate-900/40 border border-slate-800 rounded-xl text-slate-500 text-xs">
        No artifacts generated yet.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900">
      <table className="w-full text-left text-xs text-slate-300">
        <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
          <tr>
            <th className="p-3">Artifact Name</th>
            <th className="p-3">Type</th>
            <th className="p-3">Size</th>
            <th className="p-3">Checksum</th>
            <th className="p-3">Created At</th>
            <th className="p-3 text-right">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {artifacts.map((art) => (
            <tr key={art.id} className="hover:bg-slate-800/40 transition-colors">
              <td className="p-3 font-semibold text-slate-100">{art.name}</td>
              <td className="p-3 uppercase text-[10px] font-mono text-cyan-400">{art.artifact_type}</td>
              <td className="p-3 font-mono">{formatSize(art.size_bytes)}</td>
              <td className="p-3 font-mono text-slate-400 truncate max-w-[120px]">{art.checksum ? art.checksum.substring(0, 12) + '...' : 'N/A'}</td>
              <td className="p-3 text-slate-400">{new Date(art.created_at).toLocaleString()}</td>
              <td className="p-3 text-right">
                <button
                  onClick={() => handleDownload(art)}
                  className="px-3 py-1 text-xs font-semibold rounded bg-cyan-950 hover:bg-cyan-900 border border-cyan-800 text-cyan-300 transition-colors"
                >
                  ⬇️ Download
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
