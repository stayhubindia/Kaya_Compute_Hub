#!/usr/bin/env python3
"""
Master CLI Entry Point for Qwen3-4B-Base Training Controller & Recovery (Phase 5.2).

Usage:
    python scripts/run_training.py --preflight
    python scripts/run_training.py --smoke-test
    python scripts/run_training.py --train
    python scripts/run_training.py --resume
    python scripts/run_training.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config import TrainingConfig
from src.training.run_controller import TrainingRunController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5.2 — Production Qwen3-4B Training Run Controller & Recovery CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Execute comprehensive 16-point training readiness preflight audit",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute single-batch forward/backward/checkpoint smoke test",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Launch full supervised fine-tuning run with hardware gating",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest valid checkpoint with fallback recovery",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate orchestration pathways and configuration without GPU",
    )
    parser.add_argument(
        "--allow-gpu-mismatch",
        action="store_true",
        help="Permit execution on CUDA devices other than NVIDIA Tesla T4",
    )
    parser.add_argument(
        "--min-free-space-gb",
        type=float,
        default=5.0,
        help="Minimum free disk capacity required to train",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[✗] Error: Training configuration file not found at: {cfg_path}")
        return 1

    try:
        config = TrainingConfig.load_from_yaml(cfg_path)
    except Exception as e:
        print(f"[✗] Error loading training configuration: {e}")
        return 1

    controller = TrainingRunController(
        config=config,
        minimum_free_space_gb=args.min_free_space_gb,
        expected_gpu_target="Tesla T4",
        allow_gpu_target_mismatch=args.allow_gpu_mismatch,
    )

    print("=" * 70)
    print("Phase 5.2: Production Training Run Controller")
    print(f"Run Identity: {controller.run_identity.run_id}")
    print(f"Dataset Version: {controller.run_identity.dataset_version}")
    print(f"Config Hash: {controller.run_identity.training_config_hash[:16]}...")
    print("=" * 70)

    # 1. Dry Run Mode
    if args.dry_run:
        dry_res = controller.execute_dry_run()
        print("\n[✓] DRY-RUN VALIDATION COMPLETE")
        print(f"  - Run ID: {dry_res['run_identity']['run_id']}")
        print(f"  - Status: {dry_res['status']}")
        print(f"  - Storage Check: {dry_res['drive_check']['message']}")
        print(f"  - Disk Check: {dry_res['disk_check']['message']} ({dry_res['disk_check']['free_gb']:.2f} GB free)")
        print(f"  - Hardware: {dry_res['hardware']['device_name'] or 'CPU'}")
        print(f"  - Message: {dry_res['message']}")
        return 0

    # 2. Preflight Mode
    if args.preflight:
        print("\n[*] Executing Preflight Readiness Audit...")
        report = controller.execute_preflight()
        print(f"\nAudit Status: {report.overall_status}")
        for g in report.gates:
            icon = "[✓]" if g.status.value == "PASS" else ("[⚠]" if g.status.value == "WARN" else "[✗]")
            print(f"  {icon} {g.name}: {g.message}")
        if report.is_training_ready:
            print("\n[✓] Preflight Passed: System is ready for training execution.")
            return 0
        else:
            print("\n[✗] Preflight Failed: Critical readiness checks failed.")
            return 1

    # 3. Hardware Gate Check for real training / smoke-test
    hw_ok, hw_msg = controller.check_hardware_gate()
    if not hw_ok:
        print(f"\n[✗] Hardware Gate Blocked: {hw_msg}")
        print("REAL TRAINING NOT READY (GPU Unavailable / Mismatch)")
        return 1

    # 4. Smoke Test Mode
    if args.smoke_test:
        print("\n[*] Running Phase 4.2 Smoke Test on GPU...")
        try:
            from src.training.sft_trainer import ProductionSFTTrainer
            trainer = ProductionSFTTrainer(config=config)
            trainer.initialize_and_audit()
            smoke_res = trainer.run_smoke_test()
            if smoke_res.success:
                print("\n[✓] Smoke Test Successful!")
                print(f"  - Loss: {smoke_res.loss:.4f}")
                print(f"  - Peak VRAM: {smoke_res.vram_peak_allocated_mb:.1f} MB")
                return 0
            else:
                print(f"\n[✗] Smoke Test Failed: {smoke_res.message}")
                return 1
        except Exception as e:
            print(f"\n[✗] Smoke Test Exception: {e}")
            return 1

    # 5. Train / Resume Mode
    if args.train or args.resume:
        print(f"\n[*] Initiating Supervised Fine-Tuning (Resume: {args.resume})...")
        success, comp_path, telemetry, message = controller.execute_production_run(resume=args.resume)
        if success:
            print("\n" + "=" * 70)
            print("[✓] PRODUCTION TRAINING COMPLETED SUCCESSFULLY!")
            print(f"  - Completion Manifest: {comp_path}")
            if telemetry:
                print(f"  - Total Steps: {telemetry.total_steps}")
                print(f"  - Total Epochs: {telemetry.total_epochs}")
                print(f"  - Best Validation Loss: {telemetry.best_validation_loss}")
                print(f"  - Best Checkpoint: {telemetry.best_checkpoint_path}")
                print(f"  - Duration: {telemetry.training_duration_seconds:.2f}s")
            print("=" * 70)
            return 0
        else:
            print("\n" + "=" * 70)
            print(f"[✗] PRODUCTION TRAINING FAILED / BLOCKED: {message}")
            print("=" * 70)
            return 1

    print("\nNo action specified. Use --preflight, --smoke-test, --train, --resume, or --dry-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
