export interface SystemEventPayload {
  [key: string]: any;
}

export interface SystemEvent {
  event_id: string;
  event_type: string;
  job_id: string | null;
  worker_id: string | null;
  timestamp: string;
  payload: SystemEventPayload;
}

export interface JobLog {
  id: string;
  job_id: string;
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  module: string;
  message: string;
}

export interface TrainingMetric {
  id: string;
  step: number;
  epoch: number;
  name: string;
  value: number;
  split: string;
  timestamp: string;
}

export interface WorkerNode {
  id: string;
  name: string;
  hostname_label: string;
  status: 'online' | 'idle' | 'busy' | 'draining' | 'unhealthy' | 'offline';
  last_heartbeat_at: string | null;
  is_stale: boolean;
  cpu_count: number;
  memory_bytes: number;
  gpu_count: number;
  gpu_model: string;
  available_gpu_slots: number;
  allocated_gpu_slots: number;
  active_jobs_count: number;
  capabilities: Record<string, any>;
}

export interface ArtifactItem {
  id: string;
  name: string;
  artifact_type: string;
  storage_uri: string;
  checksum: string;
  size_bytes: number;
  created_at: string;
}
