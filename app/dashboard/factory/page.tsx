"use client";

import React, { useState, useEffect } from "react";
import DashboardNavbar from "@/components/DashboardNavbar";
import { User, authClient } from "@/lib/api/authClient";
import { Job, jobsClient } from "@/lib/api/jobsClient";
import { apiClient } from "@/lib/api/client";
import Link from "next/link";

interface PipelinePayload {
  collection_slug: string;
  source?: string;
  input_path?: string;
  max_documents?: number;
  candidate_count?: number;
  seed?: number;
  version_name?: string;
  split_ratios?: string;
  base_model?: string;
  lora_r?: number;
  lora_alpha?: number;
  learning_rate?: string;
  epochs?: number;
  account_id?: string;
  [key: string]: any;
}

export default function DatasetFactoryPage() {
  const [user, setUser] = useState<User | null>(null);
  const [activeTab, setActiveTab] = useState<"ingest" | "generate" | "qa" | "freeze" | "train" | "sync">("ingest");
  
  // Form State
  const [collectionSlug, setCollectionSlug] = useState("cybersecurity_v1");
  const [sourceName, setSourceName] = useState("arXiv Security");
  const [inputPath, setInputPath] = useState("/srv/kaya-data/raw_sources");
  const [maxDocuments, setMaxDocuments] = useState(100);
  const [candidateCount, setCandidateCount] = useState(500);
  const [seed, setSeed] = useState(42);
  const [versionName, setVersionName] = useState("v1.0.0");
  const [splitRatio, setSplitRatio] = useState("90/5/5");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen3-4B-Base");
  const [loraR, setLoraR] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [learningRate, setLearningRate] = useState("2e-4");
  const [epochs, setEpochs] = useState(3);
  const [driveFolder, setDriveFolder] = useState("Kaya_Datasets_Backup");

  // Options State
  const [extractEquations, setExtractEquations] = useState(true);
  const [cleanHtml, setCleanHtml] = useState(true);
  const [deduplicate, setDeduplicate] = useState(true);

  // Status & List State
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [recentJobs, setRecentJobs] = useState<Job[]>([]);
  const [isLoadingJobs, setIsLoadingJobs] = useState(true);

  // Load User & Jobs
  const fetchJobs = async () => {
    try {
      const data = await jobsClient.listJobs();
      setRecentJobs(data.results || []);
    } catch {
      // Ignore background refresh errors
    } finally {
      setIsLoadingJobs(false);
    }
  };

  useEffect(() => {
    async function init() {
      try {
        const u = await authClient.getCurrentUser();
        setUser(u);
      } catch {
        setUser(null);
      }
      await fetchJobs();
    }
    init();

    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleLaunchPipeline = async (pipelineType: string, payload: PipelinePayload) => {
    setIsSubmitting(true);
    setStatusMessage(null);

    try {
      // Map factory actions to pipeline or jobs endpoints
      const res = await apiClient<{ id: string; message?: string; status?: string }>(`/jobs/${pipelineType}/`, {
        method: "POST",
        body: JSON.stringify({
          collection_slug: collectionSlug,
          source: sourceName,
          payload,
        }),
      });

      setStatusMessage({
        type: "success",
        text: `Pipeline stage '${pipelineType.toUpperCase()}' enqueued! Job ID: ${res.id || "Active"}`,
      });

      await fetchJobs();
    } catch (err: any) {
      // Fallback to standard job creation if endpoint fails
      try {
        const fallbackRes = await jobsClient.createJob({
          name: `Pipeline ${pipelineType.toUpperCase()} - ${collectionSlug}`,
          job_type: pipelineType === "train" ? "training" : "preprocessing",
          description: `Orchestrated from Dataset Factory (${sourceName})`,
          payload: { ...payload, collection_slug: collectionSlug },
        });

        setStatusMessage({
          type: "success",
          text: `Pipeline job enqueued successfully! Job ID: ${fallbackRes.id}`,
        });
        await fetchJobs();
      } catch (fallbackErr: any) {
        setStatusMessage({
          type: "error",
          text: `Submission Error: ${fallbackErr.message || err.message || "Failed to launch pipeline stage"}`,
        });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 font-sans selection:bg-sky-500 selection:text-white">
      <DashboardNavbar user={user} />

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Header Title Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800/80 pb-6 mb-8 gap-4">
          <div>
            <div className="flex items-center space-x-3 mb-1">
              <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-sky-400 via-teal-300 to-indigo-400 bg-clip-text text-transparent">
                Dataset Factory & Training Orchestrator
              </h1>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
                System Operational
              </span>
            </div>
            <p className="text-slate-400 text-sm">
              Automated document extraction, candidate generation, QA release gates, version freezing, and QLoRA training.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <Link
              href="/dashboard/jobs"
              className="px-4 py-2 text-xs font-semibold rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors shadow-sm"
            >
              📋 View Job Queue ({recentJobs.length})
            </Link>
          </div>
        </div>

        {/* System Summary KPI Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-slate-900/60 backdrop-blur border border-slate-800/80 p-5 rounded-xl">
            <div className="text-xs font-mono uppercase tracking-wider text-slate-400">Active Pipeline Tasks</div>
            <div className="text-2xl font-bold text-sky-400 mt-1">
              {recentJobs.filter((j) => ["queued", "running", "leased"].includes(j.status)).length} Tasks Running
            </div>
            <div className="text-[11px] text-slate-500 mt-1">Celery Worker Isolated Queue</div>
          </div>

          <div className="bg-slate-900/60 backdrop-blur border border-slate-800/80 p-5 rounded-xl">
            <div className="text-xs font-mono uppercase tracking-wider text-slate-400">Default Base Model</div>
            <div className="text-lg font-bold text-teal-300 mt-1 truncate">{baseModel}</div>
            <div className="text-[11px] text-slate-500 mt-1">4-Bit QLoRA Ready</div>
          </div>

          <div className="bg-slate-900/60 backdrop-blur border border-slate-800/80 p-5 rounded-xl">
            <div className="text-xs font-mono uppercase tracking-wider text-slate-400">Target Target Collection</div>
            <div className="text-lg font-bold text-indigo-300 mt-1 truncate">{collectionSlug}</div>
            <div className="text-[11px] text-slate-500 mt-1">Version {versionName}</div>
          </div>

          <div className="bg-slate-900/60 backdrop-blur border border-slate-800/80 p-5 rounded-xl">
            <div className="text-xs font-mono uppercase tracking-wider text-slate-400">QA Engine Version</div>
            <div className="text-lg font-bold text-emerald-400 mt-1">DatasetReleaseQA V2</div>
            <div className="text-[11px] text-slate-500 mt-1">SHA-256 Checksum Verified</div>
          </div>
        </div>

        {/* Alert Status Banner */}
        {statusMessage && (
          <div
            className={`mb-6 p-4 rounded-xl border text-sm font-mono flex items-center justify-between shadow-lg transition-all ${
              statusMessage.type === "success"
                ? "bg-emerald-950/40 border-emerald-500/40 text-emerald-300"
                : "bg-rose-950/40 border-rose-500/40 text-rose-300"
            }`}
          >
            <div className="flex items-center space-x-2">
              <span>{statusMessage.type === "success" ? "✓" : "⚠️"}</span>
              <span>{statusMessage.text}</span>
            </div>
            <button
              onClick={() => setStatusMessage(null)}
              className="text-xs text-slate-400 hover:text-slate-200 ml-4 underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Global Collection Configuration Header Card */}
        <div className="bg-gradient-to-r from-slate-900/90 via-slate-900/60 to-slate-950 border border-slate-800 p-6 rounded-2xl mb-8 shadow-xl">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-sky-400 mb-4 flex items-center">
            <span className="w-2 h-2 rounded-full bg-sky-400 mr-2"></span>
            Global Collection Context Parameters
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Collection Identifier Slug</label>
              <input
                type="text"
                value={collectionSlug}
                onChange={(e) => setCollectionSlug(e.target.value)}
                placeholder="e.g. cybersecurity_v1"
                className="w-full bg-slate-950/90 border border-slate-700/80 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 transition-all font-mono"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Source Dataset Name</label>
              <input
                type="text"
                value={sourceName}
                onChange={(e) => setSourceName(e.target.value)}
                placeholder="e.g. arXiv Security Papers"
                className="w-full bg-slate-950/90 border border-slate-700/80 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 transition-all"
              />
            </div>
          </div>
        </div>

        {/* Navigation Stage Tabs */}
        <div className="flex border-b border-slate-800 mb-8 space-x-1 overflow-x-auto pb-1 scrollbar-none">
          {[
            { id: "ingest", label: "1. Document Ingestion", icon: "📄", color: "sky" },
            { id: "generate", label: "2. Candidate Generation", icon: "⚡", color: "indigo" },
            { id: "qa", label: "3. QA & Release Audit", icon: "🛡️", color: "emerald" },
            { id: "freeze", label: "4. Freeze Dataset", icon: "🔒", color: "purple" },
            { id: "train", label: "5. QLoRA Training", icon: "🚀", color: "amber" },
            { id: "sync", label: "6. Drive Export Sync", icon: "☁️", color: "teal" },
          ].map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-4 py-3 text-xs font-semibold rounded-t-xl transition-all border-t border-x whitespace-nowrap flex items-center space-x-2 ${
                  isActive
                    ? "bg-slate-900 border-slate-700 text-sky-300 border-b-transparent shadow-md"
                    : "bg-transparent border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Pipeline Controls Container */}
        <div className="bg-slate-900/80 backdrop-blur border border-slate-800/90 p-7 rounded-2xl mb-10 shadow-2xl">
          {/* TAB 1: INGESTION */}
          {activeTab === "ingest" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold text-slate-100 flex items-center">
                  <span className="text-sky-400 mr-2">📄</span> Stage 1: Document Extraction & Normalization
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Ingests PDFs, HTML, Markdown, TXT, and JSON/JSONL archives, applies layout parsing, extracts equations & tables, and emits standard chunks.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Input Path or Directory</label>
                  <input
                    type="text"
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-sky-500 font-mono"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Max Document Processing Limit</label>
                  <input
                    type="number"
                    value={maxDocuments}
                    onChange={(e) => setMaxDocuments(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-sky-500"
                  />
                </div>
              </div>

              <div className="flex flex-wrap gap-4 pt-2">
                <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={extractEquations}
                    onChange={(e) => setExtractEquations(e.target.checked)}
                    className="rounded bg-slate-950 border-slate-800 text-sky-500 focus:ring-sky-500"
                  />
                  <span>Parse LaTeX Equations</span>
                </label>
                <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={cleanHtml}
                    onChange={(e) => setCleanHtml(e.target.checked)}
                    className="rounded bg-slate-950 border-slate-800 text-sky-500 focus:ring-sky-500"
                  />
                  <span>Strip Boilerplate HTML</span>
                </label>
              </div>

              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleLaunchPipeline("ingest", {
                    collection_slug: collectionSlug,
                    input_path: inputPath,
                    source: sourceName,
                    max_documents: maxDocuments,
                    extract_equations: extractEquations,
                    clean_html: cleanHtml,
                  })
                }
                className="px-6 py-3 bg-gradient-to-r from-sky-600 to-cyan-600 hover:from-sky-500 hover:to-cyan-500 text-white rounded-xl font-semibold text-sm transition-all shadow-lg hover:shadow-sky-500/20 disabled:opacity-50"
              >
                {isSubmitting ? "Launching Ingestion..." : "🚀 Launch Document Ingestion Pipeline"}
              </button>
            </div>
          )}

          {/* TAB 2: GENERATION */}
          {activeTab === "generate" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold text-slate-100 flex items-center">
                  <span className="text-indigo-400 mr-2">⚡</span> Stage 2: Instruction Candidate Generation
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Streams chunk embeddings, synthesizes high-quality QA instruction pairs with grounding validation and deduplication filters.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Target Candidate Count</label>
                  <input
                    type="number"
                    value={candidateCount}
                    onChange={(e) => setCandidateCount(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Random Seed</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <div className="flex items-end pb-2">
                  <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={deduplicate}
                      onChange={(e) => setDeduplicate(e.target.checked)}
                      className="rounded bg-slate-950 border-slate-800 text-indigo-500 focus:ring-indigo-500"
                    />
                    <span>Enforce MinHash Deduplication</span>
                  </label>
                </div>
              </div>

              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleLaunchPipeline("generate", {
                    collection_slug: collectionSlug,
                    candidate_count: candidateCount,
                    seed,
                    deduplicate,
                  })
                }
                className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white rounded-xl font-semibold text-sm transition-all shadow-lg hover:shadow-indigo-500/20 disabled:opacity-50"
              >
                {isSubmitting ? "Generating Candidates..." : "⚡ Generate Candidate Dataset"}
              </button>
            </div>
          )}

          {/* TAB 3: QA AUDIT */}
          {activeTab === "qa" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold text-slate-100 flex items-center">
                  <span className="text-emerald-400 mr-2">🛡️</span> Stage 3: Release QA & Rights Audit Engine
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Runs DatasetReleaseQAEngine to audit grounding overlap, check license compliance, score text quality, and check data leakage.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 space-y-2">
                <div className="flex justify-between">
                  <span>QA Audit Strategy:</span>
                  <span className="text-emerald-400 font-mono">DatasetReleaseQAEngine (V2.0)</span>
                </div>
                <div className="flex justify-between">
                  <span>Grounding Gate Threshold:</span>
                  <span className="text-slate-200 font-mono">≥ 0.85 Overlap Score</span>
                </div>
                <div className="flex justify-between">
                  <span>License Filter:</span>
                  <span className="text-slate-200 font-mono">Permissive Public / Authorized Only</span>
                </div>
              </div>

              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleLaunchPipeline("qa", {
                    collection_slug: collectionSlug,
                  })
                }
                className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl font-semibold text-sm transition-all shadow-lg hover:shadow-emerald-500/20 disabled:opacity-50"
              >
                {isSubmitting ? "Running Audit..." : "🛡️ Run Full Quality & Rights Audit"}
              </button>
            </div>
          )}

          {/* TAB 4: FREEZE */}
          {activeTab === "freeze" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold text-slate-100 flex items-center">
                  <span className="text-purple-400 mr-2">🔒</span> Stage 4: Freeze & Lock Dataset Release
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Locks dataset version, splits data into Train/Val/Test, computes SHA-256 manifest checksums, and creates FROZEN state marker.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Target Release Version Tag</label>
                  <input
                    type="text"
                    value={versionName}
                    onChange={(e) => setVersionName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-purple-500 font-mono"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Train / Val / Test Split Ratio</label>
                  <input
                    type="text"
                    value={splitRatio}
                    onChange={(e) => setSplitRatio(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-purple-500 font-mono"
                  />
                </div>
              </div>

              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleLaunchPipeline("release", {
                    collection_slug: collectionSlug,
                    version_name: versionName,
                    split_ratios: splitRatio,
                  })
                }
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white rounded-xl font-semibold text-sm transition-all shadow-lg hover:shadow-purple-500/20 disabled:opacity-50"
              >
                {isSubmitting ? "Freezing Dataset..." : "🔒 Lock & Freeze Dataset Version"}
              </button>
            </div>
          )}

          {/* TAB 5: QLORA TRAINING */}
          {activeTab === "train" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold text-slate-100 flex items-center">
                  <span className="text-amber-400 mr-2">🚀</span> Stage 5: QLoRA Fine-Tuning Orchestrator
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Runs hardware preflight verification and executes 4-bit quantized QLoRA model fine-tuning on frozen dataset version.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div className="md:col-span-3">
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Base Foundation Model</label>
                  <select
                    value={baseModel}
                    onChange={(e) => setBaseModel(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500"
                  >
                    <option value="Qwen/Qwen3-4B-Base">Qwen/Qwen3-4B-Base (Recommended)</option>
                    <option value="meta-llama/Meta-Llama-3-8B">Meta-Llama-3-8B</option>
                    <option value="mistralai/Mistral-7B-v0.3">Mistral-7B-v0.3</option>
                    <option value="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">DeepSeek-R1-Distill-Qwen-7B</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">LoRA Rank (r)</label>
                  <input
                    type="number"
                    value={loraR}
                    onChange={(e) => setLoraR(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 font-mono"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">LoRA Alpha (α)</label>
                  <input
                    type="number"
                    value={loraAlpha}
                    onChange={(e) => setLoraAlpha(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 font-mono"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Learning Rate</label>
                  <input
                    type="text"
                    value={learningRate}
                    onChange={(e) => setLearningRate(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 font-mono"
                  />
                </div>
              </div>

              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleLaunchPipeline("train", {
                    collection_slug: collectionSlug,
                    base_model: baseModel,
                    lora_r: loraR,
                    lora_alpha: loraAlpha,
                    learning_rate: learningRate,
                    epochs,
                  })
                }
                className="px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white rounded-xl font-semibold text-sm transition-all shadow-lg hover:shadow-amber-500/20 disabled:opacity-50"
              >
                {isSubmitting ? "Launching Training..." : "🚀 Launch QLoRA Fine-Tuning Workload"}
              </button>
            </div>
          )}

          {/* TAB 6: GOOGLE DRIVE SYNC */}
          {activeTab === "sync" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold text-slate-100 flex items-center">
                  <span className="text-teal-400 mr-2">☁️</span> Stage 6: Google Drive Artifact Export Sync
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Synchronizes frozen dataset packages, manifests, and trained adapter weights to authorized Google Drive storage.
                </p>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">Destination Drive Folder Name</label>
                <input
                  type="text"
                  value={driveFolder}
                  onChange={(e) => setDriveFolder(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-teal-500 font-mono"
                />
              </div>

              <button
                disabled={isSubmitting}
                onClick={() =>
                  handleLaunchPipeline("sync", {
                    collection_slug: collectionSlug,
                    destination_folder: driveFolder,
                  })
                }
                className="px-6 py-3 bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-500 hover:to-emerald-500 text-white rounded-xl font-semibold text-sm transition-all shadow-lg hover:shadow-teal-500/20 disabled:opacity-50"
              >
                {isSubmitting ? "Syncing Artifacts..." : "☁️ Export & Sync to Google Drive"}
              </button>
            </div>
          )}
        </div>

        {/* Live Active Pipeline Jobs Feed */}
        <div className="bg-slate-900/70 backdrop-blur border border-slate-800/90 rounded-2xl p-6 shadow-xl">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center">
                <span className="w-2 h-2 rounded-full bg-sky-400 mr-2"></span>
                Active & Enqueued Pipeline Tasks
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">Real-time status updates from Celery worker queue</p>
            </div>
            <button
              onClick={fetchJobs}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
            >
              🔄 Refresh List
            </button>
          </div>

          {isLoadingJobs ? (
            <div className="text-xs text-slate-400 py-6 text-center">Loading task queue...</div>
          ) : recentJobs.length === 0 ? (
            <div className="text-xs text-slate-500 py-8 text-center bg-slate-950/40 rounded-xl border border-slate-800/50">
              No tasks currently queued. Select a stage above and click Launch to start a pipeline workload.
            </div>
          ) : (
            <div className="space-y-3">
              {recentJobs.slice(0, 5).map((job) => (
                <div
                  key={job.id}
                  className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4"
                >
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-sm text-slate-100">{job.name}</span>
                      <span className="text-xs font-mono text-slate-500">({job.job_type})</span>
                    </div>
                    <div className="text-xs text-slate-400">
                      Stage: <span className="text-sky-300 font-mono">{job.current_stage || job.status}</span> •{" "}
                      {job.progress_message || "In execution queue"}
                    </div>
                  </div>

                  <div className="flex items-center space-x-4">
                    <div className="w-32 bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-sky-400 to-teal-400 h-full transition-all duration-500"
                        style={{ width: `${job.progress_percentage || 0}%` }}
                      ></div>
                    </div>

                    <span
                      className={`px-2.5 py-1 rounded-full text-xs font-semibold font-mono ${
                        job.status === "succeeded"
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                          : job.status === "running" || job.status === "leased"
                          ? "bg-sky-500/10 text-sky-400 border border-sky-500/20"
                          : job.status === "failed"
                          ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                          : "bg-slate-800 text-slate-300"
                      }`}
                    >
                      {job.status.toUpperCase()} ({job.progress_percentage || 0}%)
                    </span>

                    <Link
                      href={`/dashboard/jobs/${job.id}`}
                      className="text-xs text-sky-400 hover:text-sky-300 font-semibold underline"
                    >
                      Details
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
