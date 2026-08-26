# Test Suite (`/tests`)

This directory houses system-wide testing tools and automated suites:

- **`e2e/`**: End-to-end browser and API automation tests for task creation, execution, and monitoring.
- **`integration/`**: Cross-service integration tests verifying Django REST API, Redis, Celery, and Docker executor interactions.
- **`security/`**: Automated security verification tests for RBAC, input sanitization, token revocation, and SSRF protection.
