"use client";

import React, { useState, useEffect } from "react";
import DashboardNavbar from "@/components/DashboardNavbar";
import { User, authClient } from "@/lib/api/authClient";
import { Job, jobsClient } from "@/lib/api/jobsClient";
import { integrationsClient, ConnectedAccount } from "@/lib/api/integrations-client";
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

  // Accounts State
  const [colabAccounts, setColabAccounts] = useState<ConnectedAccount[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState<string>("");

  // Form State
  const [collectionSlug, setCollectionSlug] = useState("cybersecurity_v1");
  const [sourceName, setSourceName] = useState("arXiv Security Papers");
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

  // Load User, Jobs & Colab Vault Accounts
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

      try {
        const accs = await integrationsClient.listConnectedAccounts();
        setColabAccounts(accs);
        if (accs.length > 0) {
          setSelectedAccountId(accs[0].id);
        }
      } catch {
        // Vault accounts fetch silent fail
      }

      await fetchJobs();
    }
    init();

    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleLaunchPipeline = async (pipelineStage: string, payload: PipelinePayload) => {
    setIsSubmitting(true);
    setStatusMessage(null);

    const jobTypeMap: Record<string, 'download' | 'extraction' | 'preprocessing' | 'notebook' | 'training' | 'evaluation'> = {
      ingest: 'extraction',
      generate: 'preprocessing',
      qa: 'evaluation',
      release: 'preprocessing',
      train: 'training',
      sync: 'notebook'
    };

    const mappedType = jobTypeMap[pipelineStage] || 'preprocessing';

    try {
      const jobRes = await jobsClient.createJob({
        name: `Factory Stage [${pipelineStage.toUpperCase()}] - ${collectionSlug}`,
        job_type: mappedType,
        description: `Dataset Factory stage execution for source '${sourceName}'`,
        payload: {
          ...payload,
          pipeline_stage: pipelineStage,
          collection_slug: collectionSlug,
          account_id: selectedAccountId || undefined
        }
      });

      setStatusMessage({
        type: "success",
        text: `🚀 Pipeline stage '${pipelineStage.toUpperCase()}' enqueued successfully! Job ID: ${jobRes.id}`,
      });

      await fetchJobs();
    } catch (err: any) {
      setStatusMessage({
        type: "error",
        text: `Submission Error: ${err.message || "Failed to launch pipeline stage"}`,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        {/* Header Title Section */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', borderBottom: '1px solid #1e293b', paddingBottom: '20px', marginBottom: '28px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '6px' }}>
              <h1 style={{ fontSize: '28px', fontWeight: '800', background: 'linear-gradient(135deg, #38bdf8 0%, #2dd4bf 50%, #818cf8 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Dataset Factory & Training Orchestrator
              </h1>
              <span style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', color: '#34d399', padding: '4px 10px', borderRadius: '20px', fontSize: '12px', fontWeight: '600', display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#34d399' }}></span>
                Operational
              </span>
            </div>
            <p style={{ color: '#94a3b8', fontSize: '14px', margin: 0 }}>
              Automated document extraction, candidate generation, QA release gates, version freezing, and QLoRA training.
            </p>
          </div>

          <Link
            href="/dashboard/jobs"
            style={{ background: '#1e293b', border: '1px solid #334155', color: '#f1f5f9', padding: '10px 18px', borderRadius: '8px', fontSize: '13px', fontWeight: '600', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '8px' }}
          >
            📋 View Job Queue ({recentJobs.length})
          </Link>
        </div>

        {/* System Summary KPI Stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '28px' }}>
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', padding: '20px', borderRadius: '12px' }}>
            <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#64748b', letterSpacing: '0.5px' }}>Active Pipeline Tasks</div>
            <div style={{ fontSize: '22px', fontWeight: '800', color: '#38bdf8', marginTop: '6px' }}>
              {recentJobs.filter((j) => ["queued", "running", "leased"].includes(j.status)).length} Running
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Celery Worker Queue</div>
          </div>

          <div style={{ background: '#0f172a', border: '1px solid #1e293b', padding: '20px', borderRadius: '12px' }}>
            <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#64748b', letterSpacing: '0.5px' }}>Default Base Model</div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#2dd4bf', marginTop: '6px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{baseModel}</div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>4-Bit QLoRA Ready</div>
          </div>

          <div style={{ background: '#0f172a', border: '1px solid #1e293b', padding: '20px', borderRadius: '12px' }}>
            <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#64748b', letterSpacing: '0.5px' }}>Target Collection</div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#818cf8', marginTop: '6px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{collectionSlug}</div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Version {versionName}</div>
          </div>

          <div style={{ background: '#0f172a', border: '1px solid #1e293b', padding: '20px', borderRadius: '12px' }}>
            <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#64748b', letterSpacing: '0.5px' }}>Colab Accounts in Vault</div>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#34d399', marginTop: '6px' }}>
              {colabAccounts.length} Active {colabAccounts.length === 1 ? 'Account' : 'Accounts'}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Direct Vault Credentials</div>
          </div>
        </div>

        {/* Alert Status Banner */}
        {statusMessage && (
          <div
            style={{
              padding: '16px 20px',
              borderRadius: '10px',
              marginBottom: '24px',
              fontSize: '14px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              background: statusMessage.type === 'success' ? 'rgba(6, 78, 59, 0.4)' : 'rgba(136, 19, 55, 0.4)',
              border: statusMessage.type === 'success' ? '1px solid #059669' : '1px solid #e11d48',
              color: statusMessage.type === 'success' ? '#6ee7b7' : '#fda4af'
            }}
          >
            <span>{statusMessage.text}</span>
            <button onClick={() => setStatusMessage(null)} style={{ background: 'transparent', border: 'none', color: '#cbd5e1', cursor: 'pointer', fontSize: '13px', textDecoration: 'underline' }}>Dismiss</button>
          </div>
        )}

        {/* Global Collection Configuration Header Card */}
        <div style={{ background: '#0f172a', border: '1px solid #1e293b', padding: '24px', borderRadius: '14px', marginBottom: '28px' }}>
          <h2 style={{ fontSize: '13px', fontWeight: '700', textTransform: 'uppercase', color: '#38bdf8', letterSpacing: '0.5px', marginBottom: '16px' }}>
            Global Collection Context & Worker Credentials
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>Collection Identifier Slug</label>
              <input
                type="text"
                value={collectionSlug}
                onChange={(e) => setCollectionSlug(e.target.value)}
                placeholder="e.g. cybersecurity_v1"
                style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#f8fafc', fontFamily: 'monospace', fontSize: '14px' }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>Source Dataset Name</label>
              <input
                type="text"
                value={sourceName}
                onChange={(e) => setSourceName(e.target.value)}
                placeholder="e.g. arXiv Security Papers"
                style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#f8fafc', fontSize: '14px' }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>Assigned Colab Account (Vault)</label>
              {colabAccounts.length > 0 ? (
                <select
                  value={selectedAccountId}
                  onChange={(e) => setSelectedAccountId(e.target.value)}
                  style={{ width: '100%', background: '#090d16', border: '1px solid #0284c7', borderRadius: '8px', padding: '10px 14px', color: '#38bdf8', fontWeight: '600', fontSize: '14px' }}
                >
                  {colabAccounts.map((acc) => (
                    <option key={acc.id} value={acc.id}>
                      {acc.email} ({acc.status.toUpperCase()})
                    </option>
                  ))}
                </select>
              ) : (
                <Link
                  href="/dashboard/settings/connections"
                  style={{ display: 'block', padding: '10px', background: 'rgba(217, 119, 6, 0.15)', border: '1px solid #d97706', color: '#fbbf24', borderRadius: '8px', fontSize: '12px', textDecoration: 'none', textAlign: 'center' }}
                >
                  ⚠️ No Colab Accounts linked. Click to Register in Vault
                </Link>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Stage Tabs */}
        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid #1e293b', marginBottom: '28px', overflowX: 'auto', paddingBottom: '2px' }}>
          {[
            { id: "ingest", label: "1. Document Ingestion", icon: "📄" },
            { id: "generate", label: "2. Candidate Generation", icon: "⚡" },
            { id: "qa", label: "3. QA & Release Audit", icon: "🛡️" },
            { id: "freeze", label: "4. Freeze Dataset", icon: "🔒" },
            { id: "train", label: "5. QLoRA Training", icon: "🚀" },
            { id: "sync", label: "6. Drive Export Sync", icon: "☁️" },
          ].map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                style={{
                  padding: '12px 18px',
                  borderRadius: '10px 10px 0 0',
                  fontSize: '13px',
                  fontWeight: '700',
                  border: isActive ? '1px solid #0284c7' : '1px solid transparent',
                  borderBottom: isActive ? '3px solid #0284c7' : 'none',
                  background: isActive ? '#0f172a' : 'transparent',
                  color: isActive ? '#38bdf8' : '#94a3b8',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Pipeline Controls Active Panel */}
        <div style={{ background: '#0f172a', border: '1px solid #1e293b', padding: '28px', borderRadius: '16px', marginBottom: '36px' }}>
          {/* TAB 1: INGESTION */}
          {activeTab === "ingest" && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  📄 Stage 1: Document Extraction & Normalization
                </h3>
                <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
                  Ingests PDFs, HTML, Markdown, TXT, and JSON/JSONL archives, applies layout parsing, extracts equations & tables, and emits standard chunks.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Input Path or Directory</label>
                  <input
                    type="text"
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff', fontFamily: 'monospace' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Max Document Processing Limit</label>
                  <input
                    type="number"
                    value={maxDocuments}
                    onChange={(e) => setMaxDocuments(Number(e.target.value))}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff' }}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', gap: '24px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#cbd5e1', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={extractEquations}
                    onChange={(e) => setExtractEquations(e.target.checked)}
                  />
                  Parse LaTeX Equations
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#cbd5e1', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={cleanHtml}
                    onChange={(e) => setCleanHtml(e.target.checked)}
                  />
                  Strip Boilerplate HTML
                </label>
              </div>

              <div>
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
                  style={{ background: 'linear-gradient(135deg, #0284c7 0%, #0891b2 100%)', color: '#fff', padding: '12px 24px', borderRadius: '10px', fontWeight: '700', border: 'none', cursor: 'pointer', fontSize: '14px', boxShadow: '0 4px 14px rgba(2,132,199,0.3)' }}
                >
                  {isSubmitting ? "Launching Ingestion..." : "🚀 Launch Document Ingestion Pipeline"}
                </button>
              </div>
            </div>
          )}

          {/* TAB 2: GENERATION */}
          {activeTab === "generate" && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  ⚡ Stage 2: Instruction Candidate Generation
                </h3>
                <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
                  Streams chunk embeddings, synthesizes high-quality QA instruction pairs with grounding validation and deduplication filters.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Target Candidate Count</label>
                  <input
                    type="number"
                    value={candidateCount}
                    onChange={(e) => setCandidateCount(Number(e.target.value))}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Random Seed</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff' }}
                  />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', paddingTop: '20px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#cbd5e1', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={deduplicate}
                      onChange={(e) => setDeduplicate(e.target.checked)}
                    />
                    Enforce MinHash Deduplication
                  </label>
                </div>
              </div>

              <div>
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
                  style={{ background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)', color: '#fff', padding: '12px 24px', borderRadius: '10px', fontWeight: '700', border: 'none', cursor: 'pointer', fontSize: '14px', boxShadow: '0 4px 14px rgba(79,70,229,0.3)' }}
                >
                  {isSubmitting ? "Generating Candidates..." : "⚡ Generate Candidate Dataset"}
                </button>
              </div>
            </div>
          )}

          {/* TAB 3: QA AUDIT */}
          {activeTab === "qa" && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  🛡️ Stage 3: Release QA & Rights Audit Engine
                </h3>
                <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
                  Runs DatasetReleaseQAEngine to audit grounding overlap, check license compliance, score text quality, and check data leakage.
                </p>
              </div>

              <div style={{ background: '#090d16', border: '1px solid #1e293b', padding: '16px', borderRadius: '10px', fontSize: '13px', color: '#cbd5e1' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span>QA Audit Strategy:</span>
                  <strong style={{ color: '#34d399' }}>DatasetReleaseQAEngine (V2.0)</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span>Grounding Gate Threshold:</span>
                  <strong>≥ 0.85 Overlap Score</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>License Filter:</span>
                  <strong>Permissive Public / Authorized Only</strong>
                </div>
              </div>

              <div>
                <button
                  disabled={isSubmitting}
                  onClick={() =>
                    handleLaunchPipeline("qa", {
                      collection_slug: collectionSlug,
                    })
                  }
                  style={{ background: 'linear-gradient(135deg, #059669 0%, #0d9488 100%)', color: '#fff', padding: '12px 24px', borderRadius: '10px', fontWeight: '700', border: 'none', cursor: 'pointer', fontSize: '14px', boxShadow: '0 4px 14px rgba(5,150,105,0.3)' }}
                >
                  {isSubmitting ? "Running Audit..." : "🛡️ Run Full Quality & Rights Audit"}
                </button>
              </div>
            </div>
          )}

          {/* TAB 4: FREEZE */}
          {activeTab === "freeze" && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  🔒 Stage 4: Freeze & Lock Dataset Release
                </h3>
                <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
                  Locks dataset version, splits data into Train/Val/Test, computes SHA-256 manifest checksums, and creates FROZEN state marker.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Target Release Version Tag</label>
                  <input
                    type="text"
                    value={versionName}
                    onChange={(e) => setVersionName(e.target.value)}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff', fontFamily: 'monospace' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Train / Val / Test Split Ratio</label>
                  <input
                    type="text"
                    value={splitRatio}
                    onChange={(e) => setSplitRatio(e.target.value)}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff', fontFamily: 'monospace' }}
                  />
                </div>
              </div>

              <div>
                <button
                  disabled={isSubmitting}
                  onClick={() =>
                    handleLaunchPipeline("release", {
                      collection_slug: collectionSlug,
                      version_name: versionName,
                      split_ratios: splitRatio,
                    })
                  }
                  style={{ background: 'linear-gradient(135deg, #9333ea 0%, #c026d3 100%)', color: '#fff', padding: '12px 24px', borderRadius: '10px', fontWeight: '700', border: 'none', cursor: 'pointer', fontSize: '14px', boxShadow: '0 4px 14px rgba(147,51,234,0.3)' }}
                >
                  {isSubmitting ? "Freezing Dataset..." : "🔒 Lock & Freeze Dataset Version"}
                </button>
              </div>
            </div>
          )}

          {/* TAB 5: QLORA TRAINING */}
          {activeTab === "train" && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  🚀 Stage 5: QLoRA Fine-Tuning Orchestrator
                </h3>
                <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
                  Runs hardware preflight verification and executes 4-bit quantized QLoRA model fine-tuning on frozen dataset version.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>
                <div style={{ gridColumn: 'span 3' }}>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Base Foundation Model</label>
                  <select
                    value={baseModel}
                    onChange={(e) => setBaseModel(e.target.value)}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff', fontSize: '14px' }}
                  >
                    <option value="Qwen/Qwen3-4B-Base">Qwen/Qwen3-4B-Base (Recommended)</option>
                    <option value="meta-llama/Meta-Llama-3-8B">Meta-Llama-3-8B</option>
                    <option value="mistralai/Mistral-7B-v0.3">Mistral-7B-v0.3</option>
                    <option value="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">DeepSeek-R1-Distill-Qwen-7B</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>LoRA Rank (r)</label>
                  <input
                    type="number"
                    value={loraR}
                    onChange={(e) => setLoraR(Number(e.target.value))}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff', fontFamily: 'monospace' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>LoRA Alpha (α)</label>
                  <input
                    type="number"
                    value={loraAlpha}
                    onChange={(e) => setLoraAlpha(Number(e.target.value))}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff', fontFamily: 'monospace' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Learning Rate</label>
                  <input
                    type="text"
                    value={learningRate}
                    onChange={(e) => setLearningRate(e.target.value)}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff', fontFamily: 'monospace' }}
                  />
                </div>
              </div>

              <div>
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
                  style={{ background: 'linear-gradient(135deg, #d97706 0%, #ea580c 100%)', color: '#fff', padding: '12px 24px', borderRadius: '10px', fontWeight: '700', border: 'none', cursor: 'pointer', fontSize: '14px', boxShadow: '0 4px 14px rgba(217,119,6,0.3)' }}
                >
                  {isSubmitting ? "Launching Training..." : "🚀 Launch QLoRA Fine-Tuning Workload"}
                </button>
              </div>
            </div>
          )}

          {/* TAB 6: GOOGLE DRIVE SYNC */}
          {activeTab === "sync" && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  ☁️ Stage 6: Google Drive Artifact Export Sync
                </h3>
                <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
                  Synchronizes frozen dataset packages, manifests, and trained adapter weights to authorized Google Drive storage.
                </p>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Destination Drive Folder Name</label>
                <input
                  type="text"
                  value={driveFolder}
                  onChange={(e) => setDriveFolder(e.target.value)}
                  style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#fff', fontFamily: 'monospace' }}
                />
              </div>

              <div>
                <button
                  disabled={isSubmitting}
                  onClick={() =>
                    handleLaunchPipeline("sync", {
                      collection_slug: collectionSlug,
                      destination_folder: driveFolder,
                    })
                  }
                  style={{ background: 'linear-gradient(135deg, #0d9488 0%, #059669 100%)', color: '#fff', padding: '12px 24px', borderRadius: '10px', fontWeight: '700', border: 'none', cursor: 'pointer', fontSize: '14px', boxShadow: '0 4px 14px rgba(13,148,136,0.3)' }}
                >
                  {isSubmitting ? "Syncing Artifacts..." : "☁️ Export & Sync to Google Drive"}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Live Active Pipeline Jobs Feed */}
        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div>
              <h2 style={{ fontSize: '16px', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                Active & Enqueued Pipeline Tasks
              </h2>
              <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>Real-time status updates from Celery worker queue</p>
            </div>
            <button
              onClick={fetchJobs}
              style={{ background: '#1e293b', border: '1px solid #334155', color: '#cbd5e1', padding: '8px 14px', borderRadius: '6px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}
            >
              🔄 Refresh List
            </button>
          </div>

          {isLoadingJobs ? (
            <div style={{ fontSize: '13px', color: '#94a3b8', padding: '24px 0', textAlign: 'center' }}>Loading task queue...</div>
          ) : recentJobs.length === 0 ? (
            <div style={{ fontSize: '13px', color: '#64748b', padding: '32px', textAlign: 'center', background: '#090d16', borderRadius: '10px', border: '1px solid #1e293b' }}>
              No tasks currently queued. Select a stage above and click Launch to start a pipeline workload.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {recentJobs.slice(0, 5).map((job) => (
                <div
                  key={job.id}
                  style={{
                    background: '#090d16',
                    border: '1px solid #1e293b',
                    borderRadius: '12px',
                    padding: '16px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '12px'
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '14px', fontWeight: '700', color: '#f8fafc' }}>{job.name}</span>
                      <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#64748b' }}>({job.job_type})</span>
                    </div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                      Stage: <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>{job.current_stage || job.status}</span> •{" "}
                      {job.progress_message || "In execution queue"}
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ width: '120px', background: '#1e293b', borderRadius: '10px', height: '8px', overflow: 'hidden' }}>
                      <div
                        style={{
                          background: 'linear-gradient(to right, #38bdf8, #2dd4bf)',
                          height: '100%',
                          width: `${job.progress_percentage || 0}%`,
                          transition: 'width 0.3s ease'
                        }}
                      ></div>
                    </div>

                    <span
                      style={{
                        padding: '4px 10px',
                        borderRadius: '20px',
                        fontSize: '11px',
                        fontWeight: '700',
                        fontFamily: 'monospace',
                        background: job.status === 'succeeded' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(2, 132, 199, 0.15)',
                        color: job.status === 'succeeded' ? '#34d399' : '#38bdf8',
                        border: job.status === 'succeeded' ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(2, 132, 199, 0.3)'
                      }}
                    >
                      {job.status.toUpperCase()} ({job.progress_percentage || 0}%)
                    </span>

                    <Link
                      href={`/dashboard/jobs/${job.id}`}
                      style={{ fontSize: '12px', color: '#38bdf8', fontWeight: '600', textDecoration: 'underline' }}
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
