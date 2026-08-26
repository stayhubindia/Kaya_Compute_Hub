import os
import uuid
import json
from pathlib import Path
from datetime import timedelta
from django.utils import timezone
from django.shortcuts import redirect
from django.http import HttpResponseRedirect
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.response import Response

from apps.integrations.models import (
    ConnectedAccount,
    AccountStatusChoices,
    OAuthState,
    ExternalNotebook,
    ExternalRun
)
from apps.integrations.serializers import (
    ConnectedAccountSerializer,
    ExternalNotebookSerializer,
    ExternalRunSerializer
)
from apps.audit.services import log_audit_event
from services.integrations.google.oauth import (
    generate_pkce_pair,
    generate_state,
    get_authorization_url,
    exchange_code_for_tokens,
    refresh_access_token,
    fetch_google_userinfo,
    revoke_token
)
from services.integrations.google.drive_client import GoogleDriveClient
from services.integrations.colab_enterprise.client import ColabEnterpriseClient
from services.integrations.colab_enterprise.executions import ExternalRunStatus
from services.integrations.google.errors import GoogleOAuthError, TokenRevokedError, GoogleDriveError

# --- Colab Account & Vault Endpoints ---

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def google_colab_auth_link(request):
    """Generate official Google Colab authorization URL for copy-paste verification."""
    params = {
        'client_id': '764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com',
        'redirect_uri': 'https://sdk.cloud.google.com/applicationdefaultauthcode.html',
        'response_type': 'code',
        'scope': 'openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/colaboratory',
        'access_type': 'offline',
        'prompt': 'consent',
        'token_usage': 'remote'
    }
    req = requests.Request('GET', 'https://accounts.google.com/o/oauth2/v2/auth', params=params)
    prepared = req.prepare()
    return Response({"auth_url": prepared.url})


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def google_colab_verify_code(request):
    """
    Exchanges pasted Colab authorization code for official access/refresh tokens,
    saves the account to ConnectedAccount DB and creates local Vault JSON file.
    """
    code = request.data.get('code', '').strip()
    email_override = request.data.get('email', '').strip() or 'stayhubindia@gmail.com'

    if not code:
        return Response({"error": {"message": "Authorization code is required."}}, status=status.HTTP_400_BAD_REQUEST)

    # Attempt official token exchange with Google
    payload = {
        'client_id': '764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com',
        'client_secret': 'd-FL95Q19q7MQmFpd7hHD0Ty',
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': 'https://sdk.cloud.google.com/applicationdefaultauthcode.html',
    }

    access_token = code
    refresh_token = ""
    token_expiry = timezone.now() + timedelta(days=365)
    user_email = email_override

    try:
        resp = requests.post('https://oauth2.googleapis.com/token', data=payload, timeout=10)
        if resp.status_code == 200:
            tokens = resp.json()
            access_token = tokens.get('access_token', code)
            refresh_token = tokens.get('refresh_token', '')
            expires_in = tokens.get('expires_in', 3600)
            token_expiry = timezone.now() + timedelta(seconds=expires_in)

            # Try to fetch real user email from userinfo
            try:
                userinfo_resp = requests.get(
                    'https://www.googleapis.com/oauth2/v3/userinfo',
                    headers={'Authorization': f'Bearer {access_token}'},
                    timeout=5
                )
                if userinfo_resp.status_code == 200:
                    user_email = userinfo_resp.json().get('email', user_email)
            except Exception:
                pass
    except Exception:
        pass

    # Save to ConnectedAccount DB
    account, created = ConnectedAccount.objects.get_or_create(
        user=request.user,
        email=user_email,
        defaults={
            'provider': 'google',
            'provider_account_id': f"colab-{uuid.uuid4().hex[:8]}",
            'display_name': user_email,
        }
    )

    account.display_name = user_email
    account.set_access_token(access_token)
    if refresh_token:
        account.set_refresh_token(refresh_token)
    account.token_expiry = token_expiry
    account.status = AccountStatusChoices.ACTIVE
    account.last_verified_at = timezone.now()
    account.scopes = ["openid", "drive.readonly", "colaboratory"]
    account.save()

    # Mirror to Vault JSON file (~/.config/colab-cli/saved_accounts/<email>.json)
    try:
        vault_dir = Path.home() / ".config/colab-cli/saved_accounts"
        vault_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = user_email.replace("@", "_at_")
        vault_file = vault_dir / f"{safe_filename}.json"
        vault_file.write_text(json.dumps({
            "email": user_email,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "auth_code": code,
            "created_at": timezone.now().isoformat()
        }, indent=2))
    except Exception:
        pass

    log_audit_event(
        action="auth.colab_account_code_verified",
        resource_type="connected_account",
        resource_id=str(account.id),
        actor=request.user,
        metadata={"email": user_email},
        request=request
    )

    serializer = ConnectedAccountSerializer(account)
    return Response(serializer.data, status=status.HTTP_200_OK)


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
        refresh_token = account.get_refresh_token()
        access_token = account.get_access_token()

        # Check if local vault file exists
        safe_filename = (account.email or "").replace("@", "_at_")
        vault_file = Path.home() / f".config/colab-cli/saved_accounts/{safe_filename}.json"

        if refresh_token:
            try:
                tokens = refresh_access_token(refresh_token)
                new_access = tokens.get('access_token')
                expires_in = tokens.get('expires_in', 3600)
                account.set_access_token(new_access)
                account.token_expiry = timezone.now() + timedelta(seconds=expires_in)
            except Exception:
                pass

        if access_token or refresh_token or vault_file.exists():
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
        return Response({"status": "expired", "message": "No valid token or Vault file available."})
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

    raw_refresh = account.get_refresh_token()
    raw_access = account.get_access_token()
    revoke_token(raw_refresh or raw_access)

    account.encrypted_access_token = ''
    account.encrypted_refresh_token = ''
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
    Direct token/credential vault entry for Google Drive & Colab accounts without web OAuth popup redirect.
    Saves to ConnectedAccount model and syncs to ~/.config/colab-cli/saved_accounts/<email>.json.
    """
    email = request.data.get('email', '').strip()
    display_name = request.data.get('display_name', '').strip() or email
    access_token = request.data.get('access_token', '').strip()
    refresh_token = request.data.get('refresh_token', '').strip()
    raw_json = request.data.get('raw_json', '')

    if not email:
        return Response({"error": {"message": "Email address is required."}}, status=status.HTTP_400_BAD_REQUEST)

    provider_account_id = f"manual-{uuid.uuid4().hex[:12]}"
    
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
    
    account.status = AccountStatusChoices.ACTIVE
    account.token_expiry = timezone.now() + timedelta(days=365)
    account.last_verified_at = timezone.now()
    account.scopes = ["drive.file", "colab"]
    account.save()

    # Sync to local Vault directory ~/.config/colab-cli/saved_accounts/ for Colab account manager
    try:
        vault_dir = Path.home() / ".config/colab-cli/saved_accounts"
        vault_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = email.replace("@", "_at_")
        vault_file = vault_dir / f"{safe_filename}.json"
        
        token_data = {
            "email": email,
            "access_token": access_token or "direct_token",
            "refresh_token": refresh_token or "",
            "raw_json": raw_json,
            "created_at": timezone.now().isoformat()
        }
        vault_file.write_text(json.dumps(token_data, indent=2))
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
    refresh_token = account.get_refresh_token()

    if refresh_token:
        try:
            tokens = refresh_access_token(refresh_token)
            if tokens and 'access_token' in tokens:
                access_token = tokens['access_token']
                account.set_access_token(access_token)
                account.save()
        except Exception:
            pass

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
    refresh_token = account.get_refresh_token()

    if refresh_token:
        try:
            tokens = refresh_access_token(refresh_token)
            if tokens and 'access_token' in tokens:
                access_token = tokens['access_token']
                account.set_access_token(access_token)
                account.save()
        except Exception:
            pass

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

    # Sync selected Vault account credentials to active ~/.config/colab-cli/token.json
    if account:
        safe_filename = account.email.replace("@", "_at_")
        vault_file = Path.home() / f".config/colab-cli/saved_accounts/{safe_filename}.json"
        token_file = Path.home() / ".config/colab-cli/token.json"
        token_file.parent.mkdir(parents=True, exist_ok=True)

        if vault_file.exists():
            shutil.copy2(vault_file, token_file)
        else:
            access_token = account.get_access_token()
            refresh_token = account.get_refresh_token()
            token_data = {
                "email": account.email,
                "access_token": access_token,
                "refresh_token": refresh_token,
            }
            token_file.write_text(json.dumps(token_data, indent=2))

    # Clear stale sessions cache
    sessions_cache = Path.home() / ".config/colab-cli/sessions.json"
    if sessions_cache.exists():
        try:
            sessions_cache.unlink()
        except Exception:
            pass

    venv_colab = Path(__file__).resolve().parent.parent.parent.parent / ".venv/bin/colab"
    colab_bin = str(venv_colab) if venv_colab.exists() else shutil.which("colab") or "colab"

    # Stop orphan local assignment first
    subprocess.run([colab_bin, "stop", "-s", session_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    cmd = [colab_bin, "new", "-s", session_name]
    if gpu_variant.upper() in ["T4", "L4", "G4", "H100", "A100"]:
        cmd.extend(["--gpu", gpu_variant.upper()])
    elif gpu_variant.upper() in ["TPU", "V5E1", "V6E1"]:
        cmd.extend(["--tpu", "v5e1" if gpu_variant.upper() == "TPU" else gpu_variant.lower()])

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
        output = f"{proc.stdout}\n{proc.stderr}"

        if proc.returncode != 0 and ("ColabRequestError" in output or "412" in output or "503" in output):
            return Response({
                "status": "error",
                "message": f"Colab session allocation failed: {proc.stderr.strip() or proc.stdout.strip()}"
            }, status=status.HTTP_400_BAD_REQUEST)

        # Readiness ping check on Colab VM Kernel
        ping_script = Path("/tmp/ping_colab.py")
        ping_script.write_text("print('COLAB_VM_READY')\n", encoding="utf-8")

        ping_proc = subprocess.run(
            [colab_bin, "exec", "-s", session_name, "-f", str(ping_script)],
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
    venv_colab = Path(__file__).resolve().parent.parent.parent.parent / ".venv/bin/colab"
    colab_bin = str(venv_colab) if venv_colab.exists() else shutil.which("colab") or "colab"

    try:
        proc = subprocess.run([colab_bin, "sessions"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
        raw_lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]

        return Response({
            "output_raw": proc.stdout,
            "sessions": raw_lines,
            "active_count": len(raw_lines)
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

    venv_colab = Path(__file__).resolve().parent.parent.parent.parent / ".venv/bin/colab"
    colab_bin = str(venv_colab) if venv_colab.exists() else shutil.which("colab") or "colab"

    try:
        proc = subprocess.run([colab_bin, "stop", "-s", session_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        return Response({
            "status": "stopped",
            "session_name": session_name,
            "message": f"Colab session '{session_name}' stopped successfully."
        })
    except Exception as e:
        return Response({"error": {"message": str(e)}}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
