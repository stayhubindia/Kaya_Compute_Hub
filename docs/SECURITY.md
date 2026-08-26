# Kaya Compute Hub - Security Specification & Threat Model

---

## 1. Authentication Architecture

Kaya Compute Hub operates as a single-user private admin panel. It uses **secure, cookie-based session authentication**. Passwords, session keys, and tokens are **NEVER stored in `localStorage` or `sessionStorage`**.

### Single Admin Rule
- Only one active admin account is permitted in PostgreSQL.
- User registration, public signups, viewer/operator roles, TOTP, and recovery codes are omitted.
- The single admin account is created via the Django management CLI:
  ```bash
  python manage.py create_admin --email admin@example.com --password "SecurePassword123!"
  ```

### Login Flow
1. Client submits credentials via `POST /api/v1/auth/login/` containing `email` and `password`.
2. Backend authenticates credentials using Django's `Argon2PasswordHasher`.
3. On invalid login, a generic error message (`Invalid email or password.`) is returned. User existence is never disclosed.
4. Upon successful authentication, Django issues `request.session.cycle_key()` to prevent Session Fixation attacks and sets an `HttpOnly` session cookie (`sessionid`).

---

## 2. Cookie & CSRF Security Controls

| Security Attribute | Value | Description |
| :--- | :--- | :--- |
| **HttpOnly** | `True` | Prevents client-side JavaScript access to authentication session cookies. |
| **Secure** | `True` (Prod) / `False` (Dev) | Enforces HTTPS-only cookie transmission in production environments. |
| **SameSite** | `Lax` | Mitigates Cross-Site Request Forgery (CSRF) for cross-site requests. |
| **CSRF Protection** | `X-CSRFToken` Header | State-changing requests (`POST`, `PUT`, `PATCH`, `DELETE`) require valid CSRF token validation. |
| **CORS Policy** | Explicit Allowlist | Wildcard `*` origins are strictly prohibited when `CORS_ALLOW_CREDENTIALS = True`. |

---

## 3. Password Security & Argon2 Hashing

- **Argon2 Primary Hasher**: Passwords are hashed using Argon2 (`argon2-cffi`), providing memory-hard protection against GPU brute-force attacks.
- **Single Active Admin Invariant**: The database layer enforces that at most one active admin user can exist. Attempting to create a second user raises a validation error.

---

## 4. Permission Controls (`IsAuthenticatedAdmin`)

All API endpoints are guarded by `IsAuthenticatedAdmin`, requiring an active, authenticated admin session.

---

## 5. Audit Event Stream

All security-sensitive operations record immutable audit records (`AuditEvent`):
- `auth.login_success` / `auth.login_failure` / `auth.logout`
- `job.create` / `job.cancel` / `job.retry`
- `download.requested` / `download.started` / `download.paused` / `download.resumed` / `download.cancelled` / `download.completed` / `download.url_rejected` / `download.checksum_mismatch` / `download.quota_rejected`
- `pipeline.created` / `processing_run.created` / `processing_run.started` / `processing_run.paused` / `processing_run.resumed` / `processing_run.cancelled` / `processing_run.succeeded` / `processing_run.failed`
- `training.created` / `training.started` / `training.paused` / `training.resumed` / `training.cancelled` / `training.succeeded` / `training.failed` / `model.registered` / `model.approved` / `model.archived`

Passwords, authorization headers, and session cookie values are explicitly excluded from audit metadata.
