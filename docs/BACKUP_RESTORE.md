# Kaya Compute Hub — Backup & Disaster Recovery Guide

## 1. Relational Database Backup (PostgreSQL)

### Daily Automated Backup
```bash
pg_dump -U kaya -h localhost kaya_db | gzip > /var/backups/kaya_db_$(date +%Y%m%d_%H%M%S).sql.gz
```

### Database Restore
```bash
gunzip -c /var/backups/kaya_db_20260826_120000.sql.gz | psql -U kaya -d kaya_db
```

---

## 2. Dataset Storage Backup

### Backing Up Frozen Datasets (`70-training-ready/`)
Frozen dataset directories contain cryptographic checksums (`checksums.sha256`). Copy or sync the dataset folder:
```bash
rsync -avzP /srv/kaya-data/<collection_slug>/70-training-ready/ /backup-drive/collections/<collection_slug>/70-training-ready/
```

### Verifying Backup Integrity
```bash
cd /backup-drive/collections/<collection_slug>/70-training-ready/
sha256sum -c checksums.sha256
```

---

## 3. Secret & Direct Credential Vault Recovery
- Connected account refresh tokens are stored encrypted in PostgreSQL using Django's Fernet-based field encryption.
- Ensure `SECRET_KEY` and `ENCRYPTION_KEY` in `.env` are safely backed up in a secure secret manager.
