import pytest
from unittest.mock import patch
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.datasets.models import Dataset, DatasetStatusChoices
from apps.pipelines.models import PipelineDefinition, ProcessingRun, ProcessingRunStatus
from apps.audit.models import AuditEvent

@pytest.mark.django_db
def test_create_pipeline_definition():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    client = APIClient()
    client.force_authenticate(user=admin)

    # Success Case
    resp = client.post('/api/v1/pipelines/', {
        'name': 'Standard Preprocessing Pipeline',
        'description': 'Clean, deduplicate, and split dataset',
        'version': '1.0.0',
        'stages': [
            {'name': 'validate_files', 'params': {}},
            {'name': 'inspect_schema', 'params': {}},
            {'name': 'normalize_text', 'params': {'remove_control_chars': True}},
            {'name': 'deduplicate', 'params': {}},
            {'name': 'generate_statistics', 'params': {}}
        ],
        'resource_policy': {
            'max_cpu_cores': 2.0,
            'max_memory_mb': 4096,
            'run_as_non_root': True
        }
    }, format='json')

    assert resp.status_code == 201
    pipeline_id = resp.json()['id']
    assert PipelineDefinition.objects.filter(id=pipeline_id).exists()

    # Rejection Case: Unsafe python code in stage params
    unsafe_resp = client.post('/api/v1/pipelines/', {
        'name': 'Malicious Pipeline',
        'stages': [
            {'name': 'validate_files', 'params': {'eval': 'exec("import os; os.system(\'ls\')")'}}
        ]
    }, format='json')

    assert unsafe_resp.status_code == 400

@pytest.mark.django_db
def test_create_and_execute_processing_run(tmp_path):
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    client = APIClient()
    client.force_authenticate(user=admin)

    # Create dummy source dataset file
    ds_file = tmp_path / "input_data.csv"
    ds_file.write_text("id,text\n1,hello\n2,world\n3,hello\n", encoding="utf-8")

    dataset = Dataset.objects.create(
        name="Input Data",
        storage_uri=str(ds_file),
        format="csv",
        size_bytes=ds_file.stat().st_size,
        status=DatasetStatusChoices.AVAILABLE,
        created_by=admin
    )

    pipeline = PipelineDefinition.objects.create(
        name="Cleaning Pipeline",
        stages=[
            {'name': 'validate_files', 'params': {}},
            {'name': 'inspect_schema', 'params': {}},
            {'name': 'deduplicate', 'params': {}},
            {'name': 'generate_statistics', 'params': {}}
        ],
        created_by=admin
    )

    # Submit Processing Run
    resp = client.post('/api/v1/processing-runs/', {
        'pipeline_id': str(pipeline.id),
        'source_dataset_id': str(dataset.id)
    }, format='json')

    assert resp.status_code == 201
    run_id = resp.json()['id']

    # Execute Processing Run Task (with eager task execution)
    from services.worker.tasks.processing_tasks import execute_processing_run
    result = execute_processing_run(run_id)

    assert result['status'] == 'succeeded'

    run_obj = ProcessingRun.objects.get(id=run_id)
    assert run_obj.status == ProcessingRunStatus.SUCCEEDED
    assert run_obj.output_dataset is not None

    # Verify Dataset Immutability: Source dataset is intact!
    assert dataset.storage_uri == str(ds_file)
    assert ds_file.exists()

    # Verify Derived Output Dataset
    output_ds = run_obj.output_dataset
    assert output_ds.parent_dataset == dataset
    assert hasattr(output_ds, 'manifest')
    assert output_ds.manifest.file_count > 0

    # Verify Audit Events
    assert AuditEvent.objects.filter(action="processing_run.succeeded", resource_id=str(run_id)).exists()

@pytest.mark.django_db
def test_processing_run_cancel_pause_resume():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    dataset = Dataset.objects.create(name="Input Data", storage_uri="/tmp/ds.csv", created_by=admin)
    pipeline = PipelineDefinition.objects.create(name="Pipe", stages=[{'name': 'validate_files', 'params': {}}], created_by=admin)

    run_obj = ProcessingRun.objects.create(pipeline=pipeline, source_dataset=dataset, status=ProcessingRunStatus.RUNNING, created_by=admin)

    client = APIClient()
    client.force_authenticate(user=admin)

    # Pause
    pause_resp = client.post(f'/api/v1/processing-runs/{run_obj.id}/pause/')
    assert pause_resp.status_code == 200
    run_obj.refresh_from_db()
    assert run_obj.status == ProcessingRunStatus.PAUSED

    # Resume
    with patch('services.worker.tasks.processing_tasks.execute_processing_run.delay') as mock_task:
        resume_resp = client.post(f'/api/v1/processing-runs/{run_obj.id}/resume/')
        assert resume_resp.status_code == 200
        run_obj.refresh_from_db()
        assert run_obj.status == ProcessingRunStatus.QUEUED
        mock_task.assert_called_once_with(str(run_obj.id))

    # Cancel
    cancel_resp = client.post(f'/api/v1/processing-runs/{run_obj.id}/cancel/')
    assert cancel_resp.status_code == 200
    run_obj.refresh_from_db()
    assert run_obj.status == ProcessingRunStatus.CANCELLED
