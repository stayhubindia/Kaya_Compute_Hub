import pytest
from unittest.mock import patch
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.datasets.models import Dataset, DatasetStatusChoices
from apps.pipelines.models import PipelineDefinition, ProcessingRun, ProcessingRunStatus
from apps.training.models import TrainingRun, TrainingRunStatus
from apps.models_registry.models import ModelVersion, ModelStatusChoices
from apps.audit.models import AuditEvent

@pytest.mark.django_db
def test_create_and_execute_training_run(tmp_path):
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    client = APIClient()
    client.force_authenticate(user=admin)

    # Create dummy dataset
    ds_file = tmp_path / "train_data.csv"
    ds_file.write_text("id,text\n1,a\n2,b\n", encoding="utf-8")

    dataset = Dataset.objects.create(
        name="Training Dataset",
        storage_uri=str(ds_file),
        format="csv",
        size_bytes=ds_file.stat().st_size,
        status=DatasetStatusChoices.AVAILABLE,
        created_by=admin
    )

    pipeline = PipelineDefinition.objects.create(name="Pipe", stages=[], created_by=admin)
    processing_run = ProcessingRun.objects.create(
        pipeline=pipeline,
        source_dataset=dataset,
        status=ProcessingRunStatus.SUCCEEDED,
        created_by=admin
    )

    # Create Training Run via API
    resp = client.post('/api/v1/training-runs/', {
        'name': 'MNIST Classifier',
        'dataset_id': str(dataset.id),
        'processing_run_id': str(processing_run.id),
        'backend': 'demo',
        'container_image': 'kaya/ml-trainer:demo',
        'configuration': {
            'epochs': 2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adamw'
        },
        'resource_policy': {
            'max_cpu_cores': 2.0,
            'max_memory_mb': 4096,
            'requested_gpus': 0
        }
    }, format='json')

    assert resp.status_code == 201
    run_id = resp.json()['id']

    # Execute Task
    from services.worker.tasks.training_tasks import execute_training_run
    result = execute_training_run(run_id)

    assert result['status'] == 'succeeded', f"Training execution failed with result: {result}"

    run_obj = TrainingRun.objects.get(id=run_id)
    assert run_obj.status == TrainingRunStatus.SUCCEEDED
    assert run_obj.output_model_uri != ""

    # Check metrics recorded
    assert run_obj.metrics.count() > 0

    # Check model registered in registry
    assert ModelVersion.objects.filter(training_run=run_obj).exists()
    model_ver = ModelVersion.objects.get(training_run=run_obj)
    assert model_ver.status == ModelStatusChoices.REGISTERED

    # Check Audit Log
    assert AuditEvent.objects.filter(action="training.succeeded", resource_id=str(run_id)).exists()

@pytest.mark.django_db
def test_model_registry_approval():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')

    model_ver = ModelVersion.objects.create(
        name="llama-2-7b-fine-tuned",
        version="v1.0.0",
        framework="pytorch",
        model_format="bin",
        checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status=ModelStatusChoices.REGISTERED,
        created_by=admin
    )

    client = APIClient()
    client.force_authenticate(user=admin)
    admin_resp = client.post(f'/api/v1/models/{model_ver.id}/approve/')
    assert admin_resp.status_code == 200

    model_ver.refresh_from_db()
    assert model_ver.status == ModelStatusChoices.APPROVED
    assert AuditEvent.objects.filter(action="model.approved", resource_id=str(model_ver.id)).exists()

@pytest.mark.django_db
def test_training_run_cancel_pause_resume_retry():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    dataset = Dataset.objects.create(name="Input Data", storage_uri="/tmp/ds.csv", created_by=admin)

    run_obj = TrainingRun.objects.create(
        name="Test Run",
        dataset=dataset,
        status=TrainingRunStatus.RUNNING,
        created_by=admin
    )

    client = APIClient()
    client.force_authenticate(user=admin)

    # Pause
    pause_resp = client.post(f'/api/v1/training-runs/{run_obj.id}/pause/')
    assert pause_resp.status_code == 200
    run_obj.refresh_from_db()
    assert run_obj.status == TrainingRunStatus.PAUSED

    # Resume
    with patch('services.worker.tasks.training_tasks.execute_training_run.delay') as mock_task:
        resume_resp = client.post(f'/api/v1/training-runs/{run_obj.id}/resume/')
        assert resume_resp.status_code == 200
        run_obj.refresh_from_db()
        assert run_obj.status == TrainingRunStatus.QUEUED
        mock_task.assert_called_once_with(str(run_obj.id))

    # Cancel
    cancel_resp = client.post(f'/api/v1/training-runs/{run_obj.id}/cancel/')
    assert cancel_resp.status_code == 200
    run_obj.refresh_from_db()
    assert run_obj.status == TrainingRunStatus.CANCELLED
