import os
import pytest
from django.utils import timezone
from apps.workers.models import Worker, WorkerStatusChoices
from services.trainer.policies import validate_training_configuration, validate_training_resource_policy, TrainingPolicyError
from services.trainer.containers.image_registry import is_approved_training_image, get_image_metadata
from services.trainer.scheduler import TrainerScheduler
from services.trainer.registry import get_trainer_backend, BACKEND_REGISTRY
from services.trainer.checkpoints import TrainingCheckpointManager, CheckpointRecoveryManager

def test_training_configuration_validation():
    valid_cfg = {
        "backend": "demo",
        "model_name": "resnet18",
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 10,
        "optimizer": "adamw",
        "scheduler": "cosine"
    }
    validated = validate_training_configuration(valid_cfg)
    assert validated["batch_size"] == 64
    assert validated["learning_rate"] == 0.001

    # Unsafe python code rejection
    with pytest.raises(TrainingPolicyError, match="Unsafe pattern"):
        validate_training_configuration({"code": "import os; os.system('ls')"})

    # Unsafe URL rejection
    with pytest.raises(TrainingPolicyError, match="Unsafe pattern"):
        validate_training_configuration({"url": "http://evil.com/payload.py"})

    # Invalid batch_size
    with pytest.raises(TrainingPolicyError, match="batch_size"):
        validate_training_configuration({"batch_size": 99999})

def test_container_image_approval():
    assert is_approved_training_image("kaya/ml-trainer:pytorch-2.2") is True
    assert is_approved_training_image("unapproved/hacker-container:latest") is False
    meta = get_image_metadata("kaya/ml-trainer:pytorch-2.2")
    assert meta["backend"] == "pytorch"

@pytest.mark.django_db
def test_cpu_and_gpu_scheduler_allocation_and_locking():
    # Create Worker with 2 GPUs
    worker = Worker.objects.create(
        name="gpu-node-01",
        hostname="gpu01.local",
        status=WorkerStatusChoices.IDLE,
        cpu_count=16,
        memory_bytes=34359738368,
        gpu_count=2,
        available_gpu_slots=2,
        allocated_gpu_slots=0,
        last_heartbeat_at=timezone.now()
    )

    # 1. Allocate GPU Job
    allocated_worker = TrainerScheduler.find_and_allocate_worker(requested_gpus=2)
    assert allocated_worker is not None
    assert allocated_worker.id == worker.id

    worker.refresh_from_db()
    assert worker.available_gpu_slots == 0
    assert worker.allocated_gpu_slots == 2
    assert worker.status == WorkerStatusChoices.BUSY

    # 2. Allocate another GPU Job (Should return None because capacity is 0)
    no_worker = TrainerScheduler.find_and_allocate_worker(requested_gpus=1)
    assert no_worker is None

    # 3. Release Capacity
    TrainerScheduler.release_worker_capacity(str(worker.id), allocated_gpus=2)
    worker.refresh_from_db()
    assert worker.available_gpu_slots == 2
    assert worker.allocated_gpu_slots == 0
    assert worker.status == WorkerStatusChoices.IDLE

@pytest.mark.django_db
def test_stale_worker_scheduling_rejection():
    # Stale Worker (last heartbeat > 60 seconds ago)
    stale_time = timezone.now() - timezone.timedelta(seconds=120)
    Worker.objects.create(
        name="stale-node",
        hostname="stale.local",
        status=WorkerStatusChoices.IDLE,
        cpu_count=8,
        gpu_count=1,
        available_gpu_slots=1,
        last_heartbeat_at=stale_time
    )

    w = TrainerScheduler.find_and_allocate_worker(requested_gpus=1)
    assert w is None  # Stale worker rejected!

@pytest.mark.django_db
def test_demo_backend_training_and_checkpoint_recovery(tmp_path, monkeypatch):
    from apps.datasets.models import Dataset
    from apps.training.models import TrainingRun
    
    dataset = Dataset.objects.create(name="Data", storage_uri=str(tmp_path / "data.csv"))
    run_obj = TrainingRun.objects.create(name="Test Run", dataset=dataset, configuration={"epochs": 2})

    backend = get_trainer_backend("demo")
    out_dir = tmp_path / "model_out"
    ckpt_dir = tmp_path / "checkpoints"

    monkeypatch.setenv("TRAINING_CHECKPOINT_ROOT", str(ckpt_dir))

    cfg = {"epochs": 2, "batch_size": 16, "learning_rate": 0.01}

    # Initial Run
    model_path, metrics = backend.train(
        run_id=str(run_obj.id),
        dataset_uri=str(tmp_path / "data.csv"),
        output_dir=str(out_dir),
        configuration=cfg
    )

    assert os.path.exists(model_path)
    assert metrics["epochs"] == 2

    # Checkpoint Recovery Test
    recovery_data = CheckpointRecoveryManager.get_latest_valid_checkpoint(str(run_obj.id))
    assert recovery_data is not None
    epoch, step, ckpt_data, ckpt_metrics = recovery_data
    assert epoch == 2

    # Corrupted Checkpoint Test
    ckpt_file = ckpt_dir / str(run_obj.id) / "checkpoint_epoch_2_step_20.json"
    ckpt_file.write_text('{"corrupted": true}', encoding="utf-8")

    # Should fall back or reject corrupted checkpoint!
    corrupted_data = CheckpointRecoveryManager.get_latest_valid_checkpoint(str(run_obj.id))
    assert corrupted_data is None or corrupted_data[0] < 2
