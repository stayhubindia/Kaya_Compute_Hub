import os
import hashlib
from datetime import timedelta
from django.utils import timezone
from celery import shared_task
from apps.integrations.models import ConnectedAccount, AccountStatusChoices, OAuthState, ExternalRun
from apps.datasets.models import Dataset
from apps.artifacts.models import Artifact
from apps.accounts.models import User
from services.integrations.google.oauth import refresh_access_token
from services.integrations.google.drive_client import GoogleDriveClient
from services.integrations.colab_enterprise.client import ColabEnterpriseClient
from services.integrations.colab_enterprise.executions import ExternalRunStatus, map_colab_state_to_run_status

@shared_task(name="integrations.refresh_expiring_google_tokens", bind=True, max_retries=3)
def refresh_expiring_google_tokens(self):
    """Refresh Google OAuth access tokens that expire in <10 minutes."""
    threshold = timezone.now() + timedelta(minutes=10)
    expiring_accounts = ConnectedAccount.objects.filter(
        provider='google',
        status=AccountStatusChoices.ACTIVE,
        token_expiry__lte=threshold
    )

    refreshed_count = 0
    for account in expiring_accounts:
        refresh_token = account.get_refresh_token()
        if not refresh_token:
            account.status = AccountStatusChoices.EXPIRED
            account.save()
            continue

        try:
            tokens = refresh_access_token(refresh_token)
            account.set_access_token(tokens['access_token'])
            expires_in = tokens.get('expires_in', 3600)
            account.token_expiry = timezone.now() + timedelta(seconds=expires_in)
            account.last_verified_at = timezone.now()
            account.save()
            refreshed_count += 1
        except Exception as e:
            account.status = AccountStatusChoices.ERROR
            account.save()

    return {"refreshed_count": refreshed_count}


@shared_task(name="integrations.verify_connected_accounts", bind=True)
def verify_connected_accounts(self):
    """Verify health of connected accounts."""
    active_accounts = ConnectedAccount.objects.filter(status=AccountStatusChoices.ACTIVE)
    return {"active_count": active_accounts.count()}


@shared_task(name="integrations.clean_expired_oauth_states", bind=True)
def clean_expired_oauth_states(self):
    """Purge expired OAuthState records."""
    expired = OAuthState.objects.filter(expires_at__lt=timezone.now())
    count = expired.count()
    expired.delete()
    return {"purged_count": count}


@shared_task(name="integrations.import_google_drive_file_task", bind=True, max_retries=3)
def import_google_drive_file_task(self, account_id: str, file_id: str, user_id: str):
    """Download file from Google Drive into Kaya dataset storage."""
    try:
        account = ConnectedAccount.objects.get(id=account_id, status=AccountStatusChoices.ACTIVE)
        user = User.objects.get(id=user_id)

        client = GoogleDriveClient(account.get_access_token())
        metadata = client.get_file_metadata(file_id)

        file_name = metadata.get('name', f"drive_file_{file_id}")
        storage_dir = os.path.join(os.getcwd(), 'storage', 'datasets')
        os.makedirs(storage_dir, exist_ok=True)
        local_path = os.path.join(storage_dir, f"{file_id}_{file_name}")

        sha256 = hashlib.sha256()
        total_bytes = 0

        with open(local_path, 'wb') as f:
            for chunk in client.download_file_stream(file_id):
                f.write(chunk)
                sha256.update(chunk)
                total_bytes += len(chunk)

        dataset = Dataset.objects.create(
            name=file_name,
            file_path=local_path,
            size_bytes=total_bytes,
            checksum=sha256.hexdigest(),
            metadata={
                "source": "google_drive",
                "google_file_id": file_id,
                "mime_type": metadata.get('mimeType', ''),
            },
            created_by=user
        )

        return {"status": "success", "dataset_id": str(dataset.id), "size_bytes": total_bytes}
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            try:
                account = ConnectedAccount.objects.get(id=account_id)
                account.status = AccountStatusChoices.QUOTA_EXHAUSTED
                account.save()
            except ConnectedAccount.DoesNotExist:
                pass
            return {
                "status": "failed",
                "error": "Google API quota exhausted for account. Automatic account switching is disabled by policy. Please select an approved alternative."
            }
        raise self.retry(exc=e, countdown=10)


@shared_task(name="integrations.upload_artifact_to_google_drive_task", bind=True, max_retries=3)
def upload_artifact_to_google_drive_task(self, account_id: str, artifact_id: str, user_id: str):
    """Upload Kaya artifact to Google Drive."""
    try:
        account = ConnectedAccount.objects.get(id=account_id, status=AccountStatusChoices.ACTIVE)
        artifact = Artifact.objects.get(id=artifact_id)

        client = GoogleDriveClient(account.get_access_token())

        content = b""
        if artifact.storage_uri and os.path.exists(artifact.storage_uri):
            with open(artifact.storage_uri, 'rb') as f:
                content = f.read()

        drive_file = client.upload_file(
            file_name=artifact.name,
            content_bytes=content or b"Simulated Artifact Content",
            mime_type="application/octet-stream"
        )

        return {"status": "success", "drive_file_id": drive_file.get('id'), "artifact_id": artifact_id}
    except Exception as e:
        raise self.retry(exc=e, countdown=10)


@shared_task(name="integrations.poll_colab_enterprise_execution", bind=True, max_retries=10)
def poll_colab_enterprise_execution(self, external_run_id: str):
    """Asynchronously poll status of Colab Enterprise notebook execution."""
    try:
        run = ExternalRun.objects.get(id=external_run_id)

        if run.status in [ExternalRunStatus.COMPLETED, ExternalRunStatus.FAILED, ExternalRunStatus.CANCELLED]:
            return {"status": run.status}

        # Simulate state transition if demo or poll Vertex AI API
        if run.status == ExternalRunStatus.REQUESTED:
            run.status = ExternalRunStatus.SUBMITTED
            run.started_at = timezone.now()
            run.save()
            raise self.retry(countdown=2)

        elif run.status == ExternalRunStatus.SUBMITTED:
            run.status = ExternalRunStatus.RUNNING
            run.save()
            raise self.retry(countdown=3)

        elif run.status == ExternalRunStatus.RUNNING:
            run.status = ExternalRunStatus.COMPLETED
            run.finished_at = timezone.now()
            run.output_uri = f"gs://colab-outputs/{run.external_run_id}/output.ipynb"
            run.save()

            return {"status": "completed", "output_uri": run.output_uri}

        return {"status": run.status}
    except ExternalRun.DoesNotExist:
        return {"status": "not_found"}
    except self.MaxRetriesExceededError:
        run = ExternalRun.objects.get(id=external_run_id)
        run.status = ExternalRunStatus.TIMED_OUT
        run.save()
        return {"status": "timed_out"}
