"use client";

import React, { useState, useEffect } from "react";
import DashboardNavbar from "@/components/DashboardNavbar";
import { User, authClient } from "@/lib/api/authClient";
import { Job, jobsClient } from "@/lib/api/jobsClient";
import { integrationsClient, ColabSession, ConnectedAccount } from "@/lib/api/integrations-client";
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
  const [activeTab, setActiveTab] = useState<"arxiv" | "ingest" | "generate" | "qa" | "freeze" | "train" | "sync">("arxiv");

  // Accounts State
  const [colabAccounts, setColabAccounts] = useState<ConnectedAccount[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState<string>("");

  // ArXiv Batch Downloader State
  const [arxivCategory, setArxivCategory] = useState("cs.AI");
  const [arxivDriveCategory, setArxivDriveCategory] = useState("CS");
  const [arxivDriveYear, setArxivDriveYear] = useState("2025");
  const [arxivDriveFormat, setArxivDriveFormat] = useState<"pdf" | "html">("pdf");
  const [arxivMonth, setArxivMonth] = useState(() => {
    const d = new Date(); d.setMonth(d.getMonth() - 1);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
  });
  const [arxivWorkers, setArxivWorkers] = useState(4);
  const [arxivDelay, setArxivDelay] = useState(1.0);
  const [arxivOutputDir, setArxivOutputDir] = useState("/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv");
  const [arxivJobId, setArxivJobId] = useState<string | null>(null);
  const [arxivStats, setArxivStats] = useState<any>(null);
  const [arxivStatus, setArxivStatus] = useState<string>("");
  const [arxivRunning, setArxivRunning] = useState(false);
  const [arxivLogs, setArxivLogs] = useState<string[]>([]);

  // Form State
  const [collectionSlug, setCollectionSlug] = useState("cybersecurity_v1");
  const [sourceName, setSourceName] = useState("arXiv Security Papers");
  const [inputPath, setInputPath] = useState("/content/drive/MyDrive/Kaya_Compute_Hub/raw_sources");
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
  const [colabSessions, setColabSessions] = useState<ColabSession[]>([]);
  const [selectedSessionName, setSelectedSessionName] = useState("");

  const arxivDriveInputPath = `/content/drive/MyDrive/Arxiv/${arxivDriveCategory}/${arxivDriveYear}/${arxivDriveFormat}`;

  const handleProcessArxivDriveDataset = () => {
    setInputPath(arxivDriveInputPath);
    setSourceName(`ArXiv ${arxivDriveCategory} ${arxivDriveYear} (${arxivDriveFormat.toUpperCase()})`);
    void handleLaunchPipeline("ingest", {
      input_path: arxivDriveInputPath,
      source: "arxiv_drive",
      max_documents: maxDocuments,
      parse_latex: extractEquations,
      clean_html: arxivDriveFormat === "html" && cleanHtml,
      collection_slug: collectionSlug,
    });
  };

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

      try {
        const sessionResponse = await integrationsClient.listColabSessions();
        const sessions = (sessionResponse.sessions || []).filter((session) => /^[A-Za-z0-9_-]{1,64}$/.test(session.name));
        setColabSessions(sessions);
        const mounted = sessions.find((session) => session.drive_mounted === true);
        setSelectedSessionName((mounted || sessions[0])?.name || "");
      } catch {
        // Factory prevents dispatch until a live session can be selected.
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

    const jobTypeMap: Record<string, string> = {
      ingest: 'ingest_documents',
      generate: 'generate_candidates',
      qa: 'run_quality_audit',
      release: 'freeze_dataset',
      train: 'train_qlora',
      sync: 'sync_to_drive'
    };

    const mappedType = jobTypeMap[pipelineStage] || 'preprocessing';

    const selectedSession = colabSessions.find((session) => session.name === selectedSessionName);
    if (!selectedAccountId || !selectedSessionName || !selectedSession || selectedSession.drive_mounted !== true) {
      setStatusMessage({ type: 'error', text: 'Select a live Colab session with Google Drive mounted before launching a Factory job.' });
      setIsSubmitting(false);
      return;
    }

    try {
      const jobRes = await jobsClient.createJob({
        name: `Factory Stage [${pipelineStage.toUpperCase()}] - ${collectionSlug}`,
        job_type: mappedType,
        description: `Dataset Factory stage execution for source '${sourceName}'`,
        payload: {
          ...payload,
          pipeline_stage: pipelineStage,
          collection_slug: collectionSlug,
          account_id: selectedAccountId,
          execution_target: 'colab',
          session_name: selectedSessionName,
          accelerator: selectedSession.accelerator,
          colab_data_root: '/content/drive/MyDrive/Kaya_Compute_Hub/data',
        },
        selected_google_account_id: selectedAccountId,
      });

      setStatusMessage({
        type: "success",
        text: `🚀 Pipeline stage '${pipelineStage.toUpperCase()}' is running on Colab session '${selectedSessionName}'. Job ID: ${jobRes.id}`,
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

  const buildArxivColabCode = () => `import pathlib, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
output_dir = pathlib.Path(${JSON.stringify(arxivOutputDir)})
output_dir.mkdir(parents=True, exist_ok=True)
query = urllib.parse.urlencode({'search_query': 'cat:' + ${JSON.stringify(arxivCategory)}, 'start': 0, 'max_results': 100, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
feed = urllib.request.urlopen('https://export.arxiv.org/api/query?' + query, timeout=60).read()
root = ET.fromstring(feed)
ns = {'atom': 'http://www.w3.org/2005/Atom'}
entries = root.findall('atom:entry', ns)
print('KAYA_PROGRESS=10|discovering|Found ' + str(len(entries)) + ' ArXiv records', flush=True)
for index, entry in enumerate(entries, 1):
    identifier = entry.findtext('atom:id', default='', namespaces=ns).rsplit('/', 1)[-1]
    pdf_url = next((link.attrib.get('href') for link in entry.findall('atom:link', ns) if link.attrib.get('title') == 'pdf'), '')
    if not pdf_url: continue
    target = output_dir / (identifier.replace('/', '_') + '.pdf')
    if not target.exists():
        try: urllib.request.urlretrieve(pdf_url, target)
        except Exception as exc: print('Download failed for ' + identifier + ': ' + str(exc), flush=True)
    print('KAYA_PROGRESS=' + str(10 + int(index * 90 / max(1, len(entries)))) + '|downloading|Downloaded ' + str(index) + '/' + str(len(entries)), flush=True)
    time.sleep(${JSON.stringify(arxivDelay)})
print('KAYA_RESULT={"output_dir": ' + repr(str(output_dir)) + ', "records": ' + str(len(entries)) + '}', flush=True)`;

  const handleStartArxivColabJob = async () => {
    const selectedSession = colabSessions.find((session) => session.name === selectedSessionName);
    if (!selectedAccountId || !selectedSession || selectedSession.drive_mounted !== true) {
      setArxivStatus('❌ Select a live Colab session with Drive mounted before downloading.');
      return;
    }
    setArxivRunning(true);
    setArxivStats(null);
    setArxivLogs([`[→] Dispatching ArXiv batch to Colab session ${selectedSessionName}`]);
    try {
      const data = await jobsClient.createJob({
        name: `ArXiv download: ${arxivCategory} / ${arxivMonth}`,
        job_type: 'custom_script',
        selected_google_account_id: selectedAccountId,
        payload: {
          execution_target: 'colab', session_name: selectedSessionName,
          accelerator: selectedSession.accelerator, timeout_seconds: 21600,
          code: buildArxivColabCode(), output_dir: arxivOutputDir,
        },
      });
      setArxivJobId(data.id);
      setArxivStatus(`✅ Running in Colab: ${data.id.slice(0, 8)}...`);
      const poll = setInterval(async () => {
        try {
          const job = await jobsClient.getJob(data.id);
          setArxivStatus(`[${job.status.toUpperCase()}] ${job.progress_percentage}% — ${job.progress_message || 'Colab runtime working...'}`);
          setArxivStats({ processed: job.progress_percentage, total: 100 });
          if (['succeeded', 'failed', 'cancelled'].includes(job.status)) {
            clearInterval(poll);
            setArxivRunning(false);
            setArxivLogs((previous) => [...previous, `[✓] Colab job finished: ${job.status}`]);
          }
        } catch { clearInterval(poll); setArxivRunning(false); }
      }, 5000);
    } catch (err: any) {
      setArxivStatus(`❌ Submission error: ${err.message}`);
      setArxivRunning(false);
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
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Colab Runtime Queue</div>
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

            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>Live Colab Runtime (Drive required)</label>
              {colabSessions.length > 0 ? (
                <select
                  value={selectedSessionName}
                  onChange={(e) => setSelectedSessionName(e.target.value)}
                  style={{ width: '100%', background: '#090d16', border: '1px solid #0284c7', borderRadius: '8px', padding: '10px 14px', color: '#38bdf8', fontWeight: '600', fontSize: '14px' }}
                >
                  {colabSessions.map((session) => (
                    <option key={session.name} value={session.name}>
                      {session.name} — {session.drive_mounted === true ? 'DRIVE MOUNTED' : 'DRIVE NOT MOUNTED'}
                    </option>
                  ))}
                </select>
              ) : (
                <Link href="/dashboard/integrations" style={{ display: 'block', padding: '10px', background: 'rgba(217, 119, 6, 0.15)', border: '1px solid #d97706', color: '#fbbf24', borderRadius: '8px', fontSize: '12px', textDecoration: 'none', textAlign: 'center' }}>
                  ⚠️ No live Colab runtime. Create and mount Drive in Integrations.
                </Link>
              )}
              <p style={{ fontSize: '11px', color: '#94a3b8', margin: '6px 0 0' }}>Compute runs in this Colab kernel; the VM only dispatches and records logs.</p>
            </div>
          </div>
        </div>

        {/* Navigation Stage Tabs */}
        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid #1e293b', marginBottom: '28px', overflowX: 'auto', paddingBottom: '2px' }}>
          {[
            { id: "arxiv", label: "📚 ArXiv Downloader", icon: "🔬" },
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
          {/* TAB 0: ARXIV BATCH DOWNLOADER */}
          {activeTab === "arxiv" && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <div style={{ background: 'rgba(14, 116, 144, 0.14)', border: '1px solid #155e75', borderRadius: '12px', padding: '18px' }}>
                <h3 style={{ fontSize: '17px', fontWeight: '700', color: '#67e8f9', margin: 0 }}>📂 Process ArXiv dataset already in Google Drive</h3>
                <p style={{ fontSize: '12px', color: '#a5f3fc', margin: '6px 0 16px', lineHeight: 1.5 }}>
                  Drive layout detected: <code>Arxiv/&lt;category&gt;/&lt;year&gt;/pdf|html</code>. Select a branch and start ingestion on the mounted Colab runtime.
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', alignItems: 'end' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: '600', color: '#cffafe', marginBottom: '6px' }}>Category folder</label>
                    <input value={arxivDriveCategory} onChange={(e) => setArxivDriveCategory(e.target.value.replace(/[^A-Za-z0-9_.-]/g, ''))} placeholder="CS" style={{ width: '100%', background: '#090d16', border: '1px solid #155e75', borderRadius: '7px', padding: '9px 11px', color: '#f8fafc', fontFamily: 'monospace' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: '600', color: '#cffafe', marginBottom: '6px' }}>Year</label>
                    <select value={arxivDriveYear} onChange={(e) => setArxivDriveYear(e.target.value)} style={{ width: '100%', background: '#090d16', border: '1px solid #155e75', borderRadius: '7px', padding: '9px 11px', color: '#f8fafc' }}>
                      <option value="2025">2025 (papers found)</option>
                      <option value="2026">2026 (currently empty)</option>
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: '600', color: '#cffafe', marginBottom: '6px' }}>Source format</label>
                    <select value={arxivDriveFormat} onChange={(e) => setArxivDriveFormat(e.target.value as "pdf" | "html")} style={{ width: '100%', background: '#090d16', border: '1px solid #155e75', borderRadius: '7px', padding: '9px 11px', color: '#f8fafc' }}>
                      <option value="pdf">PDF</option>
                      <option value="html">HTML</option>
                    </select>
                  </div>
                  <button type="button" disabled={isSubmitting || !arxivDriveCategory} onClick={handleProcessArxivDriveDataset} style={{ background: isSubmitting ? '#334155' : '#0891b2', color: '#fff', border: 'none', borderRadius: '8px', padding: '11px 14px', fontWeight: '700', cursor: isSubmitting ? 'not-allowed' : 'pointer' }}>
                    🚀 Start Drive Ingestion in Colab
                  </button>
                </div>
                <div style={{ marginTop: '12px', fontFamily: 'monospace', fontSize: '11px', color: '#67e8f9' }}>{arxivDriveInputPath}</div>
              </div>

              <div>
                <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
                  🔬 ArXiv Research Paper Batch Downloader
                </h3>
                <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '6px', lineHeight: '1.6' }}>
                  Discover and download complete research papers (HTML + PDF) from any ArXiv category and month.
                  Uses the same polite, retry-safe engine as <code style={{ background: '#1e293b', padding: '2px 6px', borderRadius: '4px', color: '#38bdf8' }}>download_Manager.py</code> — now integrated into Kaya via background jobs.
                </p>
              </div>

              {/* Config Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>ArXiv Category</label>
                  <input
                    type="text"
                    value={arxivCategory}
                    onChange={(e) => setArxivCategory(e.target.value)}
                    placeholder="e.g. cs.AI, astro-ph, quant-ph"
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#f8fafc', fontFamily: 'monospace', fontSize: '14px' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>Month (YYYY-MM)</label>
                  <input
                    type="text"
                    value={arxivMonth}
                    onChange={(e) => setArxivMonth(e.target.value)}
                    placeholder="e.g. 2025-01"
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#f8fafc', fontFamily: 'monospace', fontSize: '14px' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>Parallel Workers (1–6)</label>
                  <input
                    type="number"
                    min={1} max={6}
                    value={arxivWorkers}
                    onChange={(e) => setArxivWorkers(Number(e.target.value))}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#f8fafc' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>Polite Delay (seconds)</label>
                  <input
                    type="number"
                    min={0} max={10} step={0.5}
                    value={arxivDelay}
                    onChange={(e) => setArxivDelay(Number(e.target.value))}
                    style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#f8fafc' }}
                  />
                </div>
              </div>

              {/* Quick category presets */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {["cs.AI","cs.LG","cs.CR","cs.CV","cs.CL","astro-ph","quant-ph","math.OC","eess.SP"].map(cat => (
                  <button
                    key={cat}
                    onClick={() => setArxivCategory(cat)}
                    style={{
                      padding: '5px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: '600', cursor: 'pointer', fontFamily: 'monospace',
                      background: arxivCategory === cat ? '#0284c722' : '#0f172a',
                      border: arxivCategory === cat ? '1px solid #0284c7' : '1px solid #334155',
                      color: arxivCategory === cat ? '#38bdf8' : '#94a3b8',
                    }}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Output Directory Field (Google Drive / Colab Standard) */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                  📁 Google Drive Storage Path
                </label>
                <input
                  type="text"
                  value={arxivOutputDir}
                  onChange={(e) => setArxivOutputDir(e.target.value)}
                  placeholder="/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv"
                  style={{ width: '100%', background: '#090d16', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', color: '#f8fafc', fontFamily: 'monospace', fontSize: '13px' }}
                />
                <p style={{ fontSize: '11px', color: '#94a3b8', marginTop: '6px', fontFamily: 'monospace' }}>
                  Target structure: <span style={{ color: '#38bdf8' }}>{arxivOutputDir}/{arxivMonth.split('-')[0] || 'YYYY'}/pdf</span> &amp; <span style={{ color: '#38bdf8' }}>html</span>
                </p>
              </div>

              {/* Action Button */}
              <button
                disabled={arxivRunning || !arxivCategory || !arxivMonth}
                onClick={handleStartArxivColabJob}
                style={{
                  background: arxivRunning ? '#334155' : 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                  color: '#fff', padding: '14px 28px', borderRadius: '10px', fontWeight: '700', border: 'none',
                  cursor: arxivRunning ? 'not-allowed' : 'pointer', fontSize: '15px',
                  boxShadow: arxivRunning ? 'none' : '0 4px 14px rgba(79,70,229,0.4)',
                  display: 'flex', alignItems: 'center', gap: '10px'
                }}
              >
                {arxivRunning ? (
                  <>⏳ Downloading ArXiv Papers — {arxivCategory} / {arxivMonth}...</>
                ) : (
                  <>🔬 Start ArXiv Batch Download ({arxivCategory} / {arxivMonth})</>
                )}
              </button>

              {/* Status Banner */}
              {arxivStatus && (
                <div style={{ background: '#090d16', border: '1px solid #334155', borderRadius: '10px', padding: '12px 16px', fontSize: '13px', fontFamily: 'monospace', color: '#38bdf8' }}>
                  {arxivStatus}
                </div>
              )}

              {/* Live Stats Grid */}
              {arxivStats && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px' }}>
                  {[
                    { k: "total",     label: "Total",     color: '#f8fafc' },
                    { k: "discovered",label: "Discovered", color: '#38bdf8' },
                    { k: "processed", label: "Processed",  color: '#818cf8' },
                    { k: "html",      label: "HTML ✓",     color: '#34d399' },
                    { k: "pdf",       label: "PDF ✓",      color: '#2dd4bf' },
                    { k: "existing",  label: "Cached",     color: '#94a3b8' },
                    { k: "failed",    label: "Failed",     color: '#f87171' },
                  ].map(({ k, label, color }) => (
                    <div key={k} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '14px', textAlign: 'center' }}>
                      <div style={{ fontSize: '22px', fontWeight: '800', color, fontVariantNumeric: 'tabular-nums' }}>{(arxivStats[k] || 0).toLocaleString()}</div>
                      <div style={{ fontSize: '11px', color: '#64748b', fontWeight: '700', textTransform: 'uppercase', marginTop: '4px' }}>{label}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Progress bar */}
              {arxivStats && (arxivStats.total || 0) > 0 && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8', marginBottom: '6px' }}>
                    <span>{Math.round((arxivStats.processed / arxivStats.total) * 100)}% Complete</span>
                    <span>ETA: {arxivStats.eta || '--'} | Speed: {arxivStats.speed || 0} papers/min</span>
                  </div>
                  <div style={{ background: '#1e293b', borderRadius: '10px', height: '10px', overflow: 'hidden' }}>
                    <div style={{
                      background: 'linear-gradient(to right, #4f46e5, #7c3aed)',
                      height: '100%',
                      width: `${Math.min(100, Math.round((arxivStats.processed / arxivStats.total) * 100))}%`,
                      transition: 'width 0.5s ease',
                      borderRadius: '10px'
                    }} />
                  </div>
                </div>
              )}

              {/* Logs */}
              {arxivLogs.length > 0 && (
                <div style={{ background: '#060a12', border: '1px solid #1e293b', borderRadius: '10px', padding: '14px', fontFamily: 'monospace', fontSize: '12px', color: '#38bdf8', maxHeight: '160px', overflowY: 'auto' }}>
                  <div style={{ color: '#475569', fontSize: '11px', fontWeight: '700', marginBottom: '8px' }}>ACTIVITY LOGS:</div>
                  {arxivLogs.map((l, i) => <div key={i} style={{ margin: '2px 0' }}>{l}</div>)}
                </div>
              )}
            </div>
          )}

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
