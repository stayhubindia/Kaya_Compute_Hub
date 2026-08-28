from django.urls import path
from apps.integrations import views

urlpatterns = [
    # Colab Vault Account Endpoints
    path('google/direct-connect/', views.google_account_direct_connect, name='google-account-direct-connect'),
    path('google/accounts/', views.google_accounts_list, name='google-accounts-list'),
    path('google/<uuid:pk>/verify/', views.google_account_verify, name='google-account-verify'),
    path('google/<uuid:pk>/disconnect/', views.google_account_disconnect, name='google-account-disconnect'),
    path('google/<uuid:pk>/revoke/', views.google_account_revoke, name='google-account-revoke'),

    # Google Drive Endpoints
    path('google/<uuid:pk>/drive/files/', views.google_drive_list_files, name='google-drive-list-files'),
    path('google/<uuid:pk>/drive/files/<str:file_id>/', views.google_drive_file_details, name='google-drive-file-details'),
    path('google/<uuid:pk>/drive/import/', views.google_drive_import_file, name='google-drive-import-file'),
    path('google/<uuid:pk>/drive/export/', views.google_drive_export_artifact, name='google-drive-export-artifact'),

    # Colab Enterprise Endpoints
    path('colab/notebooks/', views.colab_list_notebooks, name='colab-list-notebooks'),
    path('colab/notebooks/<uuid:pk>/run/', views.colab_run_notebook, name='colab-run-notebook'),
    path('colab/runs/<uuid:pk>/', views.colab_run_details, name='colab-run-details'),
    path('colab/runs/<uuid:pk>/cancel/', views.colab_run_cancel, name='colab-run-cancel'),

    # Colab Native CLI Session Allocator Endpoints
    path('colab/authorize/start/', views.colab_authorization_start, name='colab-authorization-start'),
    path('colab/authorize/complete/', views.colab_authorization_complete, name='colab-authorization-complete'),
    path('colab/sessions/create/', views.colab_session_create, name='colab-session-create'),
    path('colab/sessions/', views.colab_sessions_list, name='colab-sessions-list'),
    path('colab/sessions/stop/', views.colab_session_stop, name='colab-session-stop'),
]
