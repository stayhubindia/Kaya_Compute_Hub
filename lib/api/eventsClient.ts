import { SystemEvent } from '../schemas/dashboard-schemas';

export interface EventStreamOptions {
  jobId?: string;
  workerId?: string;
  lastEventId?: string;
  onEvent: (event: SystemEvent) => void;
  onError?: (error: any) => void;
}

export function connectEventStream(options: EventStreamOptions): () => void {
  const params = new URLSearchParams();
  if (options.jobId) params.append('job_id', options.jobId);
  if (options.workerId) params.append('worker_id', options.workerId);
  if (options.lastEventId) params.append('last_event_id', options.lastEventId);

  const url = `/api/v1/events/stream/?${params.toString()}`;
  const eventSource = new EventSource(url, { withCredentials: true });

  const eventTypes = [
    'job.queued', 'job.leased', 'job.started', 'job.progress',
    'job.log', 'job.checkpoint', 'job.succeeded', 'job.failed',
    'job.cancelled', 'job.paused', 'job.resumed', 'worker.heartbeat',
    'worker.status_changed', 'artifact.created'
  ];

  eventTypes.forEach((evtType) => {
    eventSource.addEventListener(evtType, (e: MessageEvent) => {
      try {
        const parsed: SystemEvent = JSON.parse(e.data);
        options.onEvent(parsed);
      } catch (err) {
        console.error('Failed to parse SSE event payload:', err);
      }
    });
  });

  eventSource.onerror = (err) => {
    if (options.onError) options.onError(err);
  };

  return () => {
    eventSource.close();
  };
}
