# Infrastructure & Deployment (`/infra`)

This directory contains configuration files and deployment manifests for host infrastructure:

- **`docker/`**: Dockerfiles and `docker-compose.yml` definitions for isolated task execution runtimes and service dependencies.
- **`nginx/`**: Reverse proxy configurations, SSL/TLS termination rules, and rate-limiting blocks.
- **`systemd/`**: Systemd unit files (`.service`) for running persistent background daemons on host VM boot.
- **`monitoring/`**: Prometheus, Grafana, and log collector monitoring configurations.
