from types import SimpleNamespace
from unittest.mock import patch

from services.worker.executors.colab_executor import run_colab_job


def test_vm_worker_allocates_and_executes_colab_job(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    account = SimpleNamespace(
        email='worker@gmail.com',
        status='active',
        scopes=['openid', 'https://www.googleapis.com/auth/colaboratory'],
        token_expiry=None,
        get_access_token=lambda: 'access-token',
        get_refresh_token=lambda: 'refresh-token',
        get_credential_json=lambda: (
            '{"token":"access-token","refresh_token":"refresh-token",'
            '"client_id":"colab-cli-client","client_secret":"colab-cli-secret",'
            '"token_uri":"https://oauth2.googleapis.com/token"}'
        ),
    )
    job = SimpleNamespace(
        id='12345678-1234-1234-1234-123456789012',
        selected_google_account=account,
        payload={
            'execution_target': 'colab',
            'session_name': 'training-t4',
            'accelerator': 'T4',
            'code': 'print("hello")',
        },
    )
    progress = []
    completed = SimpleNamespace(returncode=0, stdout='hello\n', stderr='')

    with patch('services.worker.executors.colab_executor.subprocess.run') as run:
        run.side_effect = [
            SimpleNamespace(returncode=0, stdout='', stderr=''),
            SimpleNamespace(returncode=0, stdout='created', stderr=''),
            completed,
        ]
        result = run_colab_job(job, lambda pct, stage, message: progress.append((pct, stage)))

    assert result['stdout'] == 'hello\n'
    assert result['session_name'] == 'training-t4'
    assert progress[-1] == (100, 'completed')
    assert run.call_args_list[1].args[0][-2:] == ['--gpu', 'T4']
    assert run.call_args_list[2].args[0][1:4] == ['exec', '-s', 'training-t4']
