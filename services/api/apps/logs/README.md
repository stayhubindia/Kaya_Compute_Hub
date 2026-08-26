# Logs Subsystem (`apps/logs`)

The **Logs Subsystem** records, sanitizes, and streams execution logs for compute tasks running inside isolated container environments on the VM.

---

## 🔒 Security & Sanitization

- **ANSI Escaping**: All terminal color/formatting codes are stripped prior to storage/rendering to prevent broken UI layouts.
- **HTML XSS Defense**: Log content is HTML-escaped (`&lt;script&gt;`) before delivery to the dashboard.
- **Secret Redaction**: Passwords, API tokens, cookies, and bearer tokens matching sensitive patterns are automatically replaced with `[REDACTED]`.

---

## 📡 Endpoints

### `GET /api/v1/jobs/<uuid>/logs/`
- **Query Parameters**:
  - `level`: Filter logs by severity (`debug`, `info`, `warning`, `error`).
  - `since`: Filter logs created after ISO-8601 timestamp.
  - `page_size`: Number of log lines to return (default 100, max 1000).
