# Events Subsystem (`apps/events`)

The **Events Subsystem** provides real-time, authenticated Server-Sent Events (SSE) streaming for job status changes, progress updates, worker heartbeats, and system telemetry across the Kaya Compute Hub operations dashboard.

---

## 🏗️ Architecture

```
Celery Worker / Django API
           │
  EventPublisher.publish_event()
           │
           ▼
    SystemEvent Model (PostgreSQL)
           │
           ▼
  GET /api/v1/events/stream/  (SSE Endpoint)
           │
           ▼
    Next.js Dashboard (`useJobEvents`)
```

---

## 📡 Endpoints

### `GET /api/v1/events/stream/`
- **Authentication**: Required (Django Session Cookie).
- **Query Parameters**:
  - `job_id`: Filter events for a specific job UUID.
  - `worker_id`: Filter events for a specific worker node UUID.
  - `last_event_id`: Resume event stream from the given event UUID.
- **Header**: `Content-Type: text/event-stream`.
- **Heartbeat Ping**: Sends `: keepalive\n\n` comments every 2 seconds to prevent client timeout.
