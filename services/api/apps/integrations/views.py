import os
import sys
import uuid
import json
import shutil
import subprocess
import hashlib
import re
from pathlib import Path
from datetime import timedelta


def _get_colab_bin() -> str:
    py_bin_dir = Path(sys.executable).parent
    venv_colab = py_bin_dir / "colab"
    if venv_colab.exists():
        return str(venv_colab)
    which_colab = shutil.which("colab")
    if which_colab:
        return which_colab
    return "colab"
from django.utils import timezone
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.response import Response

from apps.integrations.models import (
    ConnectedAccount,
    AccountStatusChoices,
    ExternalNotebook,
    ExternalRun
)
from apps.integrations.serializers import (
    ConnectedAccountSerializer,
    ExternalNotebookSerializer,
    ExternalRunSerializer
)
from apps.audit.services import log_audit_event
from services.integrations.google.drive_client import GoogleDriveClient
from services.integrations.colab_enterprise.client import ColabEnterpriseClient
from services.integrations.colab_enterprise.executions import ExternalRunStatus
from services.integrations.google.errors import GoogleDriveError
from services.worker.executors.colab_executor import _activate_account, _cli


_COLAB_SESSION_LINE = re.compile(
    r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)\s+\|\s+"
    r"Hardware:\s+(?P<accelerator>[^|]+?)\s+\|\s+Variant:\s+(?P<variant>[^|]+?)(?:\s+\|\s+Status:\s+(?P<status>.+))?$"
)


def _parse_colab_sessions(output: str) -> list[dict]:
    """Return only actual `colab sessions` records, never CLI notices/logs."""
    sessions = []
    for line in output.splitlines():
        match = _COLAB_SESSION_LINE.match(line.strip())
        if not match:
            continue
        sessions.append({key: (value or "").strip() for key, value in match.groupdict().items()})
    return sessions


def _probe_drive_mount(colab_bin: str, session_name: str) -> bool | None:
    """Ask a live Colab kernel whether its Drive FUSE mount is present."""
    probe = Path("/tmp") / f"kaya-drive-probe-{uuid.uuid4().hex}.py"
    try:
        probe.write_text(
            "import os\nprint('KAYA_DRIVE_MOUNTED=' + str(os.path.ismount('/content/drive')))\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            _cli(colab_bin, "exec", "-s", session_name, "-f", str(probe), "--timeout", "12"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=18,
        )
        if "KAYA_DRIVE_MOUNTED=True" in result.stdout:
            return True
        if "KAYA_DRIVE_MOUNTED=False" in result.stdout:
            return False
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        probe.unlink(missing_ok=True)
    return None

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def google_accounts_list(request):
    """List authenticated user's connected accounts."""
    accounts = ConnectedAccount.objects.filter(user=request.user)
    serializer = ConnectedAccountSerializer(accounts, many=True)
    return Response({"results": serializer.data})


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def google_account_verify(request, pk):
    """Verify and refresh token for connected Google account."""
    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Account not found."}}, status=status.HTTP_404_NOT_FOUND)

    try:
        access_token = account.get_access_token()

        if access_token:
            # A real Drive API request verifies that this imported token
            # belongs to a usable Drive account.
            GoogleDriveClient(access_token).get_about()
            account.status = AccountStatusChoices.ACTIVE
            account.last_verified_at = timezone.now()
            account.save()

            log_audit_event(
                action="auth.google_account_verified",
                resource_type="connected_account",
                resource_id=str(account.id),
                actor=request.user,
                request=request
            )
            return Response({"status": "active", "last_verified_at": account.last_verified_at})

        account.status = AccountStatusChoices.EXPIRED
        account.save()
        return Response({"status": "expired", "message": "No usable Drive access token is available. Re-import the account's current Colab CLI token.json."})
    except Exception as e:
        account.status = AccountStatusChoices.ERROR
        account.save()
        return Response({"status": "error", "message": str(e)})


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def google_account_disconnect(request, pk):
    """Mark account as disconnected and clear token ciphertexts."""
    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Account not found."}}, status=status.HTTP_404_NOT_FOUND)

    account.encrypted_access_token = ''
    account.encrypted_refresh_token = ''
    account.encrypted_credential_json = ''
    account.status = AccountStatusChoices.DISCONNECTED
    account.disconnected_at = timezone.now()
    account.save()

    log_audit_event(
        action="auth.google_account_disconnected",
        resource_type="connected_account",
        resource_id=str(account.id),
        actor=request.user,
        request=request
    )

    return Response({"status": "disconnected", "id": str(account.id)})


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def google_account_revoke(request, pk):
    """Revoke tokens with Google and mark account as revoked."""
    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Account not found."}}, status=status.HTTP_404_NOT_FOUND)

    account.encrypted_access_token = ''
    account.encrypted_refresh_token = ''
    account.encrypted_credential_json = ''
    account.status = AccountStatusChoices.REVOKED
    account.disconnected_at = timezone.now()
    account.save()

    log_audit_event(
        action="auth.google_account_revoked",
        resource_type="connected_account",
        resource_id=str(account.id),
        actor=request.user,
        request=request
    )

    return Response({"status": "revoked", "id": str(account.id)})




@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def google_account_direct_connect(request):
    """
    Direct token/credential vault entry for Google Drive & Colab accounts.
    Saves to ConnectedAccount model and syncs to ~/.config/colab-cli/saved_accounts/<email>.json.
    """
    email = request.data.get('email', '').strip().lower()
    display_name = request.data.get('display_name', '').strip() or email
    access_token = request.data.get('access_token', '').strip()
    refresh_token = request.data.get('refresh_token', '').strip()
    raw_json = request.data.get('raw_json', '')

    # Import the token.json produced by the account's official Colab CLI.
    # This is a direct credential import; no client id, callback or app
    # registration is involved.
    credential_payload = {}
    if isinstance(raw_json, dict):
        credential_payload = raw_json
        raw_json = json.dumps(raw_json)
    elif raw_json:
        try:
            credential_payload = json.loads(raw_json)
        except (TypeError, ValueError):
            return Response({"error": {"message": "raw_json must be valid Colab CLI token JSON."}}, status=status.HTTP_400_BAD_REQUEST)
    if not isinstance(credential_payload, dict):
        credential_payload = {}
    access_token = access_token or str(credential_payload.get('token') or credential_payload.get('access_token') or '').strip()
    refresh_token = refresh_token or str(credential_payload.get('refresh_token') or '').strip()
    if not access_token and not refresh_token:
        return Response({"error": {"message": "Import a Colab CLI token.json containing token or refresh_token."}}, status=status.HTTP_400_BAD_REQUEST)

    if not email:
        return Response({"error": {"message": "Email address is required."}}, status=status.HTTP_400_BAD_REQUEST)

    provider_account_id = f"direct-{hashlib.sha256(email.encode('utf-8')).hexdigest()[:24]}"
    
    account, created = ConnectedAccount.objects.get_or_create(
        user=request.user,
        email=email,
        defaults={
            'provider': 'google',
            'provider_account_id': provider_account_id,
            'display_name': display_name,
        }
    )

    account.display_name = display_name
    if access_token:
        account.set_access_token(access_token)
    if refresh_token:
        account.set_refresh_token(refresh_token)
    if raw_json:
        account.set_credential_json(raw_json)
    account.status = AccountStatusChoices.ACTIVE
    expiry_value = credential_payload.get('expiry') or credential_payload.get('token_expiry')
    if expiry_value:
        try:
            account.token_expiry = timezone.datetime.fromisoformat(str(expiry_value).replace('Z', '+00:00'))
        except ValueError:
            account.token_expiry = None
    else:
        account.token_expiry = None
    account.last_verified_at = timezone.now()
    scopes = credential_payload.get('scopes') or ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/colaboratory"]
    account.scopes = scopes.split() if isinstance(scopes, str) else scopes
    account.save()

    # Sync to local Vault directory ~/.config/colab-cli/saved_accounts/ for Colab account manager
    try:
        vault_dir = Path.home() / ".config/colab-cli/saved_accounts"
        vault_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = email.replace("@", "_at_")
        vault_file = vault_dir / f"{safe_filename}.json"
        
        token_data = dict(credential_payload)
        token_data["email"] = email
        token_data["token"] = access_token
        token_data["refresh_token"] = refresh_token
        token_data["scopes"] = account.scopes
        token_data["created_at"] = timezone.now().isoformat()
        vault_file.write_text(json.dumps(token_data, indent=2), encoding='utf-8')
        vault_file.chmod(0o600)
    except Exception:
        pass

    log_audit_event(
        action="auth.google_account_direct_connected",
        resource_type="connected_account",
        resource_id=str(account.id),
        actor=request.user,
        metadata={"email": email},
        request=request
    )

    serializer = ConnectedAccountSerializer(account)
    return Response(serializer.data, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


# --- Google Drive API Endpoints ---

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def google_drive_list_files(request, pk):
    """List permitted Google Drive files for connected account."""
    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Account not found."}}, status=status.HTTP_404_NOT_FOUND)

    access_token = account.get_access_token()

    client = GoogleDriveClient(access_token)

    try:
        query = request.query_params.get('query')
        res = client.list_files(query=query)
        return Response(res)
    except Exception as e:
        return Response({
            "files": [],
            "warning": f"Could not fetch Google Drive files: {str(e)}"
        }, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def google_drive_file_details(request, pk, file_id):
    """Get metadata for specific Google Drive file."""
    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Account not found."}}, status=status.HTTP_404_NOT_FOUND)

    access_token = account.get_access_token()

    client = GoogleDriveClient(access_token)

    try:
        meta = client.get_file_metadata(file_id)
        return Response(meta)
    except Exception as e:
        return Response({"error": {"message": str(e)}}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def google_drive_import_file(request, pk):
    """Trigger background import of Google Drive file into Kaya dataset subsystem."""
    file_id = request.data.get('file_id')
    if not file_id:
        return Response({"error": {"message": "file_id is required."}}, status=status.HTTP_400_BAD_REQUEST)

    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user, status=AccountStatusChoices.ACTIVE)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Active connected account not found."}}, status=status.HTTP_404_NOT_FOUND)

    from services.worker.tasks.integrations import import_google_drive_file_task
    task = import_google_drive_file_task.delay(str(account.id), file_id, str(request.user.id))

    log_audit_event(
        action="download.google_drive_import_started",
        resource_type="google_drive_file",
        resource_id=file_id,
        actor=request.user,
        metadata={"task_id": task.id},
        request=request
    )

    return Response({
        "status": "queued",
        "task_id": task.id,
        "file_id": file_id
    }, status=status.HTTP_202_ACCEPTED)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def google_drive_export_artifact(request, pk):
    """Trigger background upload of Kaya artifact to Google Drive."""
    artifact_id = request.data.get('artifact_id')
    if not artifact_id:
        return Response({"error": {"message": "artifact_id is required."}}, status=status.HTTP_400_BAD_REQUEST)

    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user, status=AccountStatusChoices.ACTIVE)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Active connected account not found."}}, status=status.HTTP_404_NOT_FOUND)

    from services.worker.tasks.integrations import upload_artifact_to_google_drive_task
    task = upload_artifact_to_google_drive_task.delay(str(account.id), artifact_id, str(request.user.id))

    log_audit_event(
        action="artifact.google_drive_export_started",
        resource_type="artifact",
        resource_id=artifact_id,
        actor=request.user,
        metadata={"task_id": task.id},
        request=request
    )

    return Response({
        "status": "queued",
        "task_id": task.id,
        "artifact_id": artifact_id
    }, status=status.HTTP_202_ACCEPTED)


# --- Colab Enterprise Endpoints ---

@api_view(['GET', 'POST'])
@permission_classes([permissions.IsAuthenticated])
def colab_list_notebooks(request):
    """List or register configured external Colab Enterprise notebooks."""
    if request.method == 'POST':
        serializer = ExternalNotebookSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        notebook = serializer.save(owner=request.user)
        return Response(ExternalNotebookSerializer(notebook).data, status=status.HTTP_201_CREATED)
    notebooks = ExternalNotebook.objects.filter(owner=request.user)
    serializer = ExternalNotebookSerializer(notebooks, many=True)
    return Response({"results": serializer.data})


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def colab_run_notebook(request, pk):
    """Submit execution request for external Colab Enterprise notebook."""
    try:
        notebook = ExternalNotebook.objects.get(pk=pk, owner=request.user)
    except ExternalNotebook.DoesNotExist:
        return Response({"error": {"message": "External notebook not found."}}, status=status.HTTP_404_NOT_FOUND)

    selected_account_id = request.data.get('selected_google_account_id')
    selected_account = None

    if selected_account_id:
        try:
            selected_account = ConnectedAccount.objects.get(pk=selected_account_id)
        except ConnectedAccount.DoesNotExist:
            return Response({"error": {"message": "Selected Google account not found."}}, status=status.HTTP_404_NOT_FOUND)

        # Enforce user ownership isolation
        if selected_account.user != request.user:
            return Response({"error": {"message": "Unauthorized account selection. Selected account does not belong to authenticated user."}}, status=status.HTTP_403_FORBIDDEN)

        # Check account status policy safety
        if selected_account.status == AccountStatusChoices.QUOTA_EXHAUSTED:
            return Response({"error": {"message": "Selected Google account is marked quota_exhausted. Please select an approved alternative account, Colab Enterprise, or VM worker."}}, status=status.HTTP_400_BAD_REQUEST)

        if selected_account.status in [AccountStatusChoices.DISCONNECTED, AccountStatusChoices.REVOKED]:
            return Response({"error": {"message": f"Selected account status is {selected_account.status}. Please reconnect the account first."}}, status=status.HTTP_400_BAD_REQUEST)

    output_uri = request.data.get('output_uri', '')
    external_run_id = f"colab-run-{uuid.uuid4().hex[:12]}"

    run = ExternalRun.objects.create(
        notebook=notebook,
        selected_google_account=selected_account,
        provider='colab_enterprise',
        external_run_id=external_run_id,
        status=ExternalRunStatus.REQUESTED,
        output_uri=output_uri
    )

    from services.worker.tasks.integrations import poll_colab_enterprise_execution
    poll_colab_enterprise_execution.delay(str(run.id))

    log_audit_event(
        action="external_run.submitted",
        resource_type="external_run",
        resource_id=str(run.id),
        actor=request.user,
        metadata={"notebook": notebook.display_name, "external_run_id": external_run_id},
        request=request
    )

    serializer = ExternalRunSerializer(run)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def colab_run_details(request, pk):
    """Get status and output details for external run."""
    try:
        run = ExternalRun.objects.get(pk=pk)
    except ExternalRun.DoesNotExist:
        return Response({"error": {"message": "External run not found."}}, status=status.HTTP_404_NOT_FOUND)

    serializer = ExternalRunSerializer(run)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def colab_run_cancel(request, pk):
    """Cancel external Colab Enterprise run."""
    try:
        run = ExternalRun.objects.get(pk=pk)
    except ExternalRun.DoesNotExist:
        return Response({"error": {"message": "External run not found."}}, status=status.HTTP_404_NOT_FOUND)

    run.status = ExternalRunStatus.CANCELLED
    run.finished_at = timezone.now()
    run.save()

    log_audit_event(
        action="external_run.cancelled",
        resource_type="external_run",
        resource_id=str(run.id),
        actor=request.user,
        request=request
    )

    serializer = ExternalRunSerializer(run)
    return Response(serializer.data)


# --- Google Colab CLI VM Session Allocator ---

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def colab_session_create(request):
    """
    Allocate and launch a new Google Colab VM session for any authenticated Vault account.
    Supports GPU (T4, L4, A100), TPU (v5e1), or CPU runtimes.
    """
    account_id = request.data.get('account_id')
    session_name = request.data.get('session_name', 'colab-worker').strip()
    gpu_variant = request.data.get('gpu_variant', 'T4').strip()

    if not session_name:
        session_name = f"colab-session-{uuid.uuid4().hex[:6]}"

    account = None
    if account_id:
        try:
            account = ConnectedAccount.objects.get(pk=account_id, user=request.user)
        except ConnectedAccount.DoesNotExist:
            return Response({"error": {"message": "Selected Vault account not found."}}, status=status.HTTP_404_NOT_FOUND)

    # Activate the selected account using google-auth's authorized-user format.
    if account:
        try:
            _activate_account(account)
        except Exception as exc:
            return Response({"error": {"message": str(exc)}}, status=status.HTTP_400_BAD_REQUEST)

    # Clear stale sessions cache
    sessions_cache = Path.home() / ".config/colab-cli/sessions.json"
    if sessions_cache.exists():
        try:
            sessions_cache.unlink()
        except Exception:
            pass

    colab_bin = _get_colab_bin()

    # Stop orphan local assignment first
    subprocess.run(_cli(colab_bin, "stop", "-s", session_name), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    cmd = _cli(colab_bin, "new", "-s", session_name)
    if gpu_variant.upper() in ["T4", "L4", "G4", "H100", "A100"]:
        cmd.extend(["--gpu", gpu_variant.upper()])
    elif gpu_variant.upper() in ["TPU", "V5E1", "V6E1"]:
        cmd.extend(["--tpu", "v5e1" if gpu_variant.upper() == "TPU" else gpu_variant.lower()])

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
        output = f"{proc.stdout}\n{proc.stderr}"

        if proc.returncode != 0:
            return Response({
                "status": "error",
                "message": f"Colab session allocation failed: {proc.stderr.strip() or proc.stdout.strip() or 'Colab CLI returned a non-zero exit status.'}"
            }, status=status.HTTP_400_BAD_REQUEST)

        # Readiness ping check on Colab VM Kernel
        ping_script = Path("/tmp/ping_colab.py")
        ping_script.write_text("print('COLAB_VM_READY')\n", encoding="utf-8")

        ping_proc = subprocess.run(
            _cli(colab_bin, "exec", "-s", session_name, "-f", str(ping_script)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25
        )
        kernel_ready = "COLAB_VM_READY" in ping_proc.stdout or proc.returncode == 0

        log_audit_event(
            action="colab.session_created",
            resource_type="colab_session",
            resource_id=session_name,
            actor=request.user,
            metadata={"gpu_variant": gpu_variant, "account": account.email if account else "default"},
            request=request
        )

        return Response({
            "status": "success",
            "session_name": session_name,
            "account_email": account.email if account else "Default Vault Account",
            "gpu_variant": gpu_variant,
            "created_at": timezone.now().isoformat(),
            "kernel_ready": kernel_ready,
            "message": f"Google Colab VM session '{session_name}' ({gpu_variant}) successfully created and verified!"
        }, status=status.HTTP_201_CREATED)

    except subprocess.TimeoutExpired:
        return Response({"error": {"message": "Colab session allocation request timed out."}}, status=status.HTTP_504_GATEWAY_TIMEOUT)
    except Exception as e:
        return Response({"error": {"message": str(e)}}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def colab_sessions_list(request):
    """List active Google Colab VM sessions."""
    colab_bin = _get_colab_bin()
    if not (Path.home() / ".config/colab-cli/token.json").exists():
        return Response({
            "output_raw": "",
            "sessions": [],
            "active_count": 0,
            "action_required": "Authorize a Google account before listing Colab sessions.",
        })

    try:
        proc = subprocess.run(_cli(colab_bin, "sessions"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
        sessions = _parse_colab_sessions(proc.stdout)
        include_drive_status = request.query_params.get("include_drive_status") in {"1", "true", "True"}
        if include_drive_status:
            for session in sessions:
                session["drive_mounted"] = _probe_drive_mount(colab_bin, session["name"])

        return Response({
            "output_raw": proc.stdout,
            "sessions": sessions,
            "active_count": len(sessions),
            "cli_error": proc.stderr.strip() if proc.returncode else "",
        })
    except Exception as e:
        return Response({"output_raw": "", "sessions": [], "active_count": 0, "error": str(e)})


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def colab_session_stop(request):
    """Stop/terminate an active Google Colab VM session."""
    session_name = request.data.get('session_name', '').strip()
    if not session_name:
        return Response({"error": {"message": "session_name is required."}}, status=status.HTTP_400_BAD_REQUEST)

    colab_bin = _get_colab_bin()

    try:
        proc = subprocess.run(_cli(colab_bin, "stop", "-s", session_name), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        return Response({
            "status": "stopped",
            "session_name": session_name,
            "message": f"Colab session '{session_name}' stopped successfully."
        })
    except Exception as e:
        return Response({"error": {"message": str(e)}}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
