"use client";

import { useState } from "react";
import DashboardNavbar from "@/components/DashboardNavbar";

interface PipelineJobPayload {
  collection_slug: string;
  input_path?: string;
  source?: string;
  max_documents?: number;
  seed?: number;
  version_name?: string;
  base_model?: string;
  account_id?: string;
}

export default function DatasetFactoryPage() {
  const [activeTab, setActiveTab] = useState<"ingest" | "generate" | "qa" | "freeze" | "train" | "sync">("ingest");
  const [collectionSlug, setCollectionSlug] = useState("cybersecurity_v1");
  const [inputPath, setInputPath] = useState("/srv/kaya-data/raw_sources");
  const [sourceName, setSourceName] = useState("arXiv");
  const [versionName, setVersionName] = useState("v1.0");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen3-4B-Base");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmitJob = async (endpoint: string, payload: PipelineJobPayload) => {
    setIsSubmitting(true);
    setStatusMessage(null);
    try {
      const res = await fetch(`/api/v1/jobs/${endpoint}/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (res.ok) {
        setStatusMessage(`Job successfully queued! Job ID: ${data.id}`);
      } else {
        setStatusMessage(`Error: ${data.detail || data.error?.message || "Failed to submit job"}`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Network error";
      setStatusMessage(`Network error submitting job: ${msg}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <DashboardNavbar user={null} />
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-6 mb-8">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-teal-400 via-sky-400 to-indigo-400 bg-clip-text text-transparent">
              Dataset Factory & Training Orchestrator
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Deterministic document ingestion, candidate generation, QA release gates, and QLoRA training.
            </p>
          </div>
        </div>

        {statusMessage && (
          <div className="mb-6 p-4 rounded-lg bg-slate-900 border border-slate-700 text-sm font-mono text-cyan-300">
            {statusMessage}
          </div>
        )}

        {/* Factory Navigation Tabs */}
        <div className="flex border-b border-slate-800 mb-8 space-x-2 overflow-x-auto">
          {[
            { id: "ingest", label: "1. Document Ingestion" },
            { id: "generate", label: "2. Candidate Generation" },
            { id: "qa", label: "3. Release QA Audit" },
            { id: "freeze", label: "4. Freeze Dataset" },
            { id: "train", label: "5. QLoRA Training" },
            { id: "sync", label: "6. Google Drive Sync" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 whitespace-nowrap ${
                activeTab === tab.id
                  ? "border-sky-400 text-sky-400 bg-slate-900/50"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Common Parameters Panel */}
        <div className="bg-slate-900/60 backdrop-blur border border-slate-800 p-6 rounded-xl mb-8">
          <h2 className="text-lg font-semibold text-slate-200 mb-4">Collection Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">Collection Slug</label>
              <input
                type="text"
                value={collectionSlug}
                onChange={(e) => setCollectionSlug(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-sky-500"
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">Source Name</label>
              <input
                type="text"
                value={sourceName}
                onChange={(e) => setSourceName(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-sky-500"
              />
            </div>
          </div>
        </div>

        {/* Tab Specific Panels */}
        <div className="bg-slate-900/60 backdrop-blur border border-slate-800 p-6 rounded-xl">
          {activeTab === "ingest" && (
            <div className="space-y-4">
              <h3 className="text-md font-semibold text-slate-200">1. Document Extraction & Ingestion</h3>
              <p className="text-xs text-slate-400">
                Discovers PDFs, HTML, Markdown, TXT, and JSON/JSONL, extracts sections & equations, and creates deterministic chunks.
              </p>
              <div>
                <label className="block text-xs font-mono text-slate-400 mb-1">Input Directory / File Path</label>
                <input
                  type="text"
                  value={inputPath}
                  onChange={(e) => setInputPath(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-sky-500"
                />
              </div>
              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleSubmitJob("ingest", {
                    collection_slug: collectionSlug,
                    input_path: inputPath,
                    source: sourceName,
                  })
                }
                className="px-5 py-2.5 bg-sky-600 hover:bg-sky-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
              >
                {isSubmitting ? "Submitting..." : "Launch Document Ingestion"}
              </button>
            </div>
          )}

          {activeTab === "generate" && (
            <div className="space-y-4">
              <h3 className="text-md font-semibold text-slate-200">2. Instruction Candidate Generation</h3>
              <p className="text-xs text-slate-400">
                Streams source chunks, synthesizes instruction candidates with grounding validation, deduplication, and quality scores.
              </p>
              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleSubmitJob("generate", {
                    collection_slug: collectionSlug,
                    source: sourceName,
                    seed: 42,
                  })
                }
                className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
              >
                {isSubmitting ? "Submitting..." : "Generate Candidates"}
              </button>
            </div>
          )}

          {activeTab === "qa" && (
            <div className="space-y-4">
              <h3 className="text-md font-semibold text-slate-200">3. Release QA & Rights Audit</h3>
              <p className="text-xs text-slate-400">
                Verifies groundings, rights classification, deduplication, and source leakage before dataset freezing.
              </p>
              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleSubmitJob("qa", {
                    collection_slug: collectionSlug,
                  })
                }
                className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
              >
                {isSubmitting ? "Submitting..." : "Run Quality Audit"}
              </button>
            </div>
          )}

          {activeTab === "freeze" && (
            <div className="space-y-4">
              <h3 className="text-md font-semibold text-slate-200">4. Freeze Dataset</h3>
              <p className="text-xs text-slate-400">
                Locks dataset version, writes dataset-manifest.json, SHA-256 checksums, and creates FROZEN marker.
              </p>
              <div>
                <label className="block text-xs font-mono text-slate-400 mb-1">Target Version Name</label>
                <input
                  type="text"
                  value={versionName}
                  onChange={(e) => setVersionName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-sky-500"
                />
              </div>
              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleSubmitJob("release", {
                    collection_slug: collectionSlug,
                    version_name: versionName,
                  })
                }
                className="px-5 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
              >
                {isSubmitting ? "Submitting..." : "Freeze & Lock Dataset"}
              </button>
            </div>
          )}

          {activeTab === "train" && (
            <div className="space-y-4">
              <h3 className="text-md font-semibold text-slate-200">5. QLoRA Model Training</h3>
              <p className="text-xs text-slate-400">
                Runs hardware preflight and launches QLoRA fine-tuning on frozen dataset version.
              </p>
              <div>
                <label className="block text-xs font-mono text-slate-400 mb-1">Base Model Name</label>
                <input
                  type="text"
                  value={baseModel}
                  onChange={(e) => setBaseModel(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-sky-500"
                />
              </div>
              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleSubmitJob("train", {
                    collection_slug: collectionSlug,
                    base_model: baseModel,
                  })
                }
                className="px-5 py-2.5 bg-amber-600 hover:bg-amber-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
              >
                {isSubmitting ? "Submitting..." : "Launch QLoRA Training"}
              </button>
            </div>
          )}

          {activeTab === "sync" && (
            <div className="space-y-4">
              <h3 className="text-md font-semibold text-slate-200">6. Google Drive Artifact Sync</h3>
              <p className="text-xs text-slate-400">
                Syncs frozen datasets, manifests, and training checkpoints to explicitly authorized Google Drive.
              </p>
              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleSubmitJob("sync", {
                    collection_slug: collectionSlug,
                    account_id: "explicit_account_id",
                  })
                }
                className="px-5 py-2.5 bg-teal-600 hover:bg-teal-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
              >
                {isSubmitting ? "Submitting..." : "Sync Collection to Drive"}
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
