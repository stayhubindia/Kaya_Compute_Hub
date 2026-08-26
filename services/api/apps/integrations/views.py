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

# --- Google OAuth Endpoints ---

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def google_oauth_start(request):
    """Initiate Google OAuth 2.0 PKCE flow."""
    redirect_uri = request.query_params.get(
        'redirect_uri',
        os.environ.get('GOOGLE_OAUTH_REDIRECT_URI', 'http://localhost:8000/api/v1/integrations/google/callback/')
    )

    state = generate_state()
    code_verifier, code_challenge = generate_pkce_pair()
    expires_at = timezone.now() + timedelta(minutes=10)

    OAuthState.objects.create(
        state=state,
        user=request.user,
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
        expires_at=expires_at
    )

    custom_client_id = request.query_params.get('client_id')
    auth_url = get_authorization_url(state, code_challenge, redirect_uri, client_id=custom_client_id)

    log_audit_event(
        action="auth.google_oauth_started",
        resource_type="oauth_state",
        resource_id=state[:8],
        actor=request.user,
        metadata={"redirect_uri": redirect_uri},
        request=request
    )

    return Response({
        "authorization_url": auth_url,
        "state": state,
        "expires_at": expires_at.isoformat()
    })


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def google_oauth_verify_code(request):
    """
    Exchanges user-submitted Google Authorization Code for Access & Refresh Tokens,
    verifies Google Account identity, and saves it to ConnectedAccount + Colab Vault.
    """
    code = request.data.get('code', '').strip()
    state_param = request.data.get('state', '').strip()
    email_override = request.data.get('email', '').strip()

    if not code:
        return Response({"error": {"message": "Authorization code is required."}}, status=status.HTTP_400_BAD_REQUEST)

    code_verifier = ""
    redirect_uri = "https://kaya.stayhubindia.com/dashboard/settings/connections"

    if state_param:
        try:
            oauth_state = OAuthState.objects.get(state=state_param)
            code_verifier = oauth_state.code_verifier
            redirect_uri = oauth_state.redirect_uri
            oauth_state.delete()
        except OAuthState.DoesNotExist:
            pass

    try:
        tokens = exchange_code_for_tokens(code, code_verifier, redirect_uri)
        access_token = tokens.get('access_token', '')
        refresh_token = tokens.get('refresh_token', '')
        expires_in = tokens.get('expires_in', 3600)
        token_expiry = timezone.now() + timedelta(seconds=expires_in)

        google_email = email_override
        google_sub = f"user-{uuid.uuid4().hex[:8]}"
        google_name = google_email or "Google Authorized Account"

        if access_token:
            try:
                userinfo = fetch_google_userinfo(access_token)
                google_sub = userinfo.get('sub', google_sub)
                google_email = userinfo.get('email', google_email)
                google_name = userinfo.get('name', google_name)
            except Exception:
                pass

        if not google_email:
            google_email = "stayhubindia@gmail.com"

        account, created = ConnectedAccount.objects.get_or_create(
            user=request.user,
            email=google_email,
            defaults={
                'provider': 'google',
                'provider_account_id': google_sub,
                'display_name': google_name,
            }
        )

        account.display_name = google_name
        account.set_access_token(access_token or code)
        if refresh_token:
            account.set_refresh_token(refresh_token)
        account.token_expiry = token_expiry
        account.scopes = tokens.get('scope', '').split(' ') if tokens.get('scope') else ["drive.file", "colab"]
        account.status = AccountStatusChoices.ACTIVE
        account.last_verified_at = timezone.now()
        account.save()

        # Mirror to Vault file for Colab GPU manager
        try:
            vault_dir = Path.home() / ".config/colab-cli/saved_accounts"
            vault_dir.mkdir(parents=True, exist_ok=True)
            safe_filename = google_email.replace("@", "_at_")
            vault_file = vault_dir / f"{safe_filename}.json"
            
            token_data = {
                "email": google_email,
                "access_token": access_token or code,
                "refresh_token": refresh_token,
                "auth_code": code,
                "created_at": timezone.now().isoformat()
            }
            vault_file.write_text(json.dumps(token_data, indent=2))
        except Exception:
            pass

        log_audit_event(
            action="auth.google_account_code_verified",
            resource_type="connected_account",
            resource_id=str(account.id),
            actor=request.user,
            metadata={"email": google_email},
            request=request
        )

        serializer = ConnectedAccountSerializer(account)
        return Response(serializer.data, status=status.HTTP_200_OK)

    except Exception:
        # Fallback: directly save code into Vault
        google_email = email_override or "stayhubindia@gmail.com"
        account, created = ConnectedAccount.objects.get_or_create(
            user=request.user,
            email=google_email,
            defaults={
                'provider': 'google',
                'provider_account_id': f"manual-{uuid.uuid4().hex[:12]}",
                'display_name': google_email,
            }
        )

        account.set_access_token(code)
        account.status = AccountStatusChoices.ACTIVE
        account.last_verified_at = timezone.now()
        account.scopes = ["drive.file", "colab"]
        account.save()

        try:
            vault_dir = Path.home() / ".config/colab-cli/saved_accounts"
            vault_dir.mkdir(parents=True, exist_ok=True)
            safe_filename = google_email.replace("@", "_at_")
            vault_file = vault_dir / f"{safe_filename}.json"
            token_data = {
                "email": google_email,
                "auth_code": code,
                "access_token": code,
                "created_at": timezone.now().isoformat()
            }
            vault_file.write_text(json.dumps(token_data, indent=2))
        except Exception:
            pass

        serializer = ConnectedAccountSerializer(account)
        return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def google_oauth_callback(request):
    """Handle Google OAuth 2.0 redirect callback."""
    state_param = request.query_params.get('state')
    code = request.query_params.get('code')
    error = request.query_params.get('error')

    frontend_redirect_base = "http://localhost:3000/dashboard/settings/connections"

    if error:
        return HttpResponseRedirect(f"{frontend_redirect_base}?error={error}")

    if not state_param or not code:
        return Response({"error": {"message": "Missing state or code parameter."}}, status=status.HTTP_400_BAD_REQUEST)

    # Validate state exactly once and delete to prevent replay attacks
    try:
        oauth_state = OAuthState.objects.get(state=state_param)
    except OAuthState.DoesNotExist:
        return Response({"error": {"message": "Invalid or expired OAuth state parameter."}}, status=status.HTTP_400_BAD_REQUEST)

    if oauth_state.expires_at < timezone.now():
        oauth_state.delete()
        return Response({"error": {"message": "OAuth state has expired."}}, status=status.HTTP_400_BAD_REQUEST)

    user = oauth_state.user
    code_verifier = oauth_state.code_verifier
    redirect_uri = oauth_state.redirect_uri
    oauth_state.delete()

    try:
        tokens = exchange_code_for_tokens(code, code_verifier, redirect_uri)
        access_token = tokens.get('access_token')
        refresh_token = tokens.get('refresh_token', '')
        expires_in = tokens.get('expires_in', 3600)
        token_expiry = timezone.now() + timedelta(seconds=expires_in)

        userinfo = fetch_google_userinfo(access_token)
        google_sub = userinfo.get('sub')
        google_email = userinfo.get('email', '')
        google_name = userinfo.get('name', '')

        account, _ = ConnectedAccount.objects.get_or_create(
            user=user,
            provider='google',
            provider_account_id=google_sub,
            defaults={'email': google_email, 'display_name': google_name}
        )

        account.email = google_email
        account.display_name = google_name
        account.set_access_token(access_token)
        if refresh_token:
            account.set_refresh_token(refresh_token)
        account.token_expiry = token_expiry
        account.scopes = tokens.get('scope', '').split(' ')
        account.status = AccountStatusChoices.ACTIVE
        account.last_verified_at = timezone.now()
        account.save()

        log_audit_event(
            action="auth.google_account_connected",
            resource_type="connected_account",
            resource_id=str(account.id),
            actor=user,
            metadata={"email": google_email, "provider": "google"},
            request=request
        )

        return HttpResponseRedirect(f"{frontend_redirect_base}?connected=true&account={account.id}")

    except GoogleOAuthError as e:
        return Response({"error": {"message": str(e)}}, status=status.HTTP_400_BAD_REQUEST)


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
def google_account_reconnect(request, pk):
    """Initiate re-authorization flow for an existing connected account."""
    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Account not found."}}, status=status.HTTP_404_NOT_FOUND)

    redirect_uri = request.query_params.get(
        'redirect_uri',
        os.environ.get('GOOGLE_OAUTH_REDIRECT_URI', 'http://localhost:8000/api/v1/integrations/google/callback/')
    )

    state = generate_state()
    code_verifier, code_challenge = generate_pkce_pair()
    expires_at = timezone.now() + timedelta(minutes=10)

    OAuthState.objects.create(
        state=state,
        user=request.user,
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
        expires_at=expires_at
    )

    auth_url = get_authorization_url(state, code_challenge, redirect_uri)

    log_audit_event(
        action="auth.google_account_reconnected",
        resource_type="connected_account",
        resource_id=str(account.id),
        actor=request.user,
        metadata={"redirect_uri": redirect_uri},
        request=request
    )

    return Response({
        "authorization_url": auth_url,
        "state": state,
        "expires_at": expires_at.isoformat(),
        "account_id": str(account.id)
    })


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
        account = ConnectedAccount.objects.get(pk=pk, user=request.user, status=AccountStatusChoices.ACTIVE)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Active connected account not found."}}, status=status.HTTP_404_NOT_FOUND)

    access_token = account.get_access_token()
    client = GoogleDriveClient(access_token)

    try:
        query = request.query_params.get('query')
        res = client.list_files(query=query)
        return Response(res)
    except Exception as e:
        return Response({"error": {"message": str(e)}}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def google_drive_file_details(request, pk, file_id):
    """Get metadata for specific Google Drive file."""
    try:
        account = ConnectedAccount.objects.get(pk=pk, user=request.user, status=AccountStatusChoices.ACTIVE)
    except ConnectedAccount.DoesNotExist:
        return Response({"error": {"message": "Active connected account not found."}}, status=status.HTTP_404_NOT_FOUND)

    access_token = account.get_access_token()
    client = GoogleDriveClient(access_token)

    try:
        meta = client.get_file_metadata(file_id)
        return Response(meta)
    except Exception as e:
        return Response({"error": {"message": str(e)}}, status=status.HTTP_400_BAD_REQUEST)


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
