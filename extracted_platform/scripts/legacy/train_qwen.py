#!/usr/bin/env python3
"""
CLI Utility for Qwen3-4B-Base QLoRA Training Readiness, Smoke Testing & Fine-Tuning (Phase 4.2).
Provides safe preflight auditing, single-batch dry-runs, smoke testing,
training manifest generation, and supervised fine-tuning execution.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config import TrainingConfig
from src.training.sft_trainer import ProductionSFTTrainer
from src.training.trainer import DryRunExecutor
from src.training.utils import TrainingManifest, compute_file_sha256, detect_hardware_environment
from src.training.validation import TrainingPreflightValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-4B-Base QLoRA Production SFT Engine CLI (Phase 4.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Execute 16-point training preflight validation audit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute single-batch forward pass to profile loss and memory consumption",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute end-to-end smoke test (forward, backward, optimizer step, val batch, checkpoint save/reload)",
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Generate and save cryptographic training_manifest.json",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Launch full QLoRA supervised fine-tuning (Requires verified hardware)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Alias for --resume checkpoint path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps before stopping",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override micro-batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    return parser


def check_real_hardware_readiness(config: TrainingConfig) -> Tuple[bool, List[str]]:
    """
    Phase 4.1.1 & 4.2 Hardware Verification:
    - NVIDIA Tesla T4 detected
    - CUDA available
    - Actual Qwen3-4B-Base path / weights present
    - Actual Qwen3 tokenizer present
    - 4-bit NF4 working
    - LoRA targets validated
    - dataset-v1.0 remains FROZEN
    """
    reasons = []
    hw = detect_hardware_environment()

    if not hw.cuda_available:
        reasons.append("CUDA is not available in current environment.")

    if not hw.device_name or ("T4" not in hw.device_name and "Tesla" not in hw.device_name):
        reasons.append(f"Target GPU 'NVIDIA Tesla T4' not detected (current: {hw.device_name or 'None'}).")

    model_path = Path(config.model.path)
    if not model_path.exists():
        reasons.append(f"Qwen3-4B-Base weights directory not found at: {model_path}")

    tok_path = Path(config.tokenizer.model_path)
    if not tok_path.exists():
        reasons.append(f"Qwen3 tokenizer directory not found at: {tok_path}")

    return len(reasons) == 0, reasons


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found at '{config_path}'", file=sys.stderr)
        return 1

    config = TrainingConfig.load_from_yaml(config_path)

    # Apply CLI overrides
    if args.epochs is not None:
        config.training.num_train_epochs = args.epochs
    if args.batch_size is not None:
        config.training.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.max_seq_length is not None:
        config.tokenizer.max_seq_length = args.max_seq_length
    if args.seed is not None:
        config.training.seed = args.seed
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir

    resume_ckpt = args.resume or args.checkpoint

    # Validate rules
    rule_errors = config.validate_rules()
    if rule_errors:
        print("❌ Configuration Rule Validation Errors:", file=sys.stderr)
        for err in rule_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    # Default action if no flag specified: run preflight audit
    if not (args.preflight or args.dry_run or args.smoke_test or args.generate_manifest or args.train):
        print("ℹ️  No action specified. Running standard preflight validation audit...\n")
        args.preflight = True

    preflight_report = None

    # 1. Preflight Audit
    if args.preflight or args.dry_run or args.generate_manifest:
        print("================================================================================")
        print("                   QWEN3-4B-BASE QLORA PREFLIGHT AUDIT                          ")
        print("================================================================================")
        validator = TrainingPreflightValidator(config)
        preflight_report = validator.run_preflight()
        print(preflight_report.to_markdown())
        print("\n================================================================================\n")

        report_save_path = Path("outputs/preflight_report.json")
        preflight_report.save_json(report_save_path)
        print(f"📄 Saved preflight report JSON to: {report_save_path}")

        if not preflight_report.is_training_ready:
            print("❌ Preflight validation FAILED. Critical readiness gates were not satisfied.")
            if not args.dry_run and not args.smoke_test:
                return 1

    # 2. Dry Run
    if args.dry_run:
        print("\n================================================================================")
        print("                   EXECUTING SINGLE-BATCH DRY RUN FORWARD PASS                  ")
        print("================================================================================")
        executor = DryRunExecutor(config)
        dry_run_result = executor.execute_dry_run()

        print(f"Status:             {'SUCCESS' if dry_run_result.success else 'FAILED'}")
        print(f"Loss:               {dry_run_result.loss:.4f}")
        print(f"Input Shape:        {dry_run_result.input_shape}")
        print(f"Batch Size:         {dry_run_result.batch_size}")
        print(f"Sequence Length:    {dry_run_result.sequence_length}")
        print(f"GPU Allocated VRAM: {dry_run_result.gpu_allocated_mb:.2f} MB")
        print(f"Peak GPU VRAM:      {dry_run_result.peak_gpu_memory_mb:.2f} MB")
        print(f"Execution Time:     {dry_run_result.execution_time_seconds:.2f}s")
        print(f"Message:            {dry_run_result.message}")
        print("================================================================================\n")

        if not dry_run_result.success:
            return 1

    # 3. Generate Training Manifest
    if args.generate_manifest or args.preflight:
        manifest_path = Path("datasets/production/manifests/training_manifest.json")
        train_file = Path(config.dataset.train_file)
        if not train_file.exists():
            train_file = Path("datasets/production/processed/train.jsonl")
        val_file = Path(config.dataset.validation_file)
        if not val_file.exists():
            val_file = Path("datasets/production/processed/validation.jsonl")
        test_file = Path(config.dataset.test_file)
        if not test_file.exists():
            test_file = Path("datasets/production/processed/test.jsonl")

        train_sha = compute_file_sha256(train_file) if train_file.exists() else ""
        val_sha = compute_file_sha256(val_file) if val_file.exists() else ""
        test_sha = compute_file_sha256(test_file) if test_file.exists() else ""

        manifest = TrainingManifest(
            manifest_version="1.0.0",
            dataset_version=config.dataset.version,
            dataset_sha256=train_sha,
            train_sha256=train_sha,
            validation_sha256=val_sha,
            test_sha256=test_sha,
            model_name=config.model.name,
            model_path=config.model.path,
            tokenizer_path=config.tokenizer.model_path,
            quantization_config=config.quantization.model_dump(),
            lora_config=config.lora.model_dump(),
            hyperparameters=config.training.model_dump(),
            hardware_environment=detect_hardware_environment().model_dump(),
            training_schedule=preflight_report.schedule_estimates if preflight_report else {},
        )
        manifest.save_json(manifest_path)
        print(f"🔒 Generated sealed training manifest at: {manifest_path}")

    # 4. Smoke Test
    if args.smoke_test:
        print("\n================================================================================")
        print("                   EXECUTING PRE-TRAINING SMOKE TEST PIPELINE                   ")
        print("================================================================================")
        trainer = ProductionSFTTrainer(config)
        smoke_result = trainer.run_smoke_test()

        print(f"Status:                    {'SUCCESS' if smoke_result.success else 'FAILED'}")
        print(f"Train Loss:                {smoke_result.loss:.4f} (Finite: {smoke_result.loss_finite})")
        print(f"Gradients Finite:          {smoke_result.gradients_finite}")
        print(f"Optimizer Step:            {smoke_result.optimizer_step_successful}")
        print(f"Validation Loss:           {smoke_result.validation_loss:.4f}")
        print(f"Checkpoint Save & Reload:  Written={smoke_result.checkpoint_written}, Reloaded={smoke_result.checkpoint_reloaded}")
        print(f"VRAM Memory Profile:       Load={smoke_result.vram_after_load_mb:.1f}MB, Batch={smoke_result.vram_after_batch_mb:.1f}MB, Peak={smoke_result.vram_peak_allocated_mb:.1f}MB")
        print(f"Duration:                  {smoke_result.duration_seconds:.2f}s")
        print(f"Message:                   {smoke_result.message}")
        print("================================================================================\n")

        if not smoke_result.success:
            print("❌ Smoke test FAILED.")
            return 1
        print("✅ SMOKE TEST PASSED — READY FOR FULL TRAINING")

    # 5. Full Training Launch
    if args.train:
        print("\n================================================================================")
        print("                   CHECKING PRODUCTION HARDWARE READINESS                       ")
        print("================================================================================")
        hw_ready, hw_reasons = check_real_hardware_readiness(config)

        if not hw_ready:
            print("\n🚨 REAL T4 TRAINING NOT READY 🚨\n")
            print("The following hardware or environment prerequisites were not met:")
            for r in hw_reasons:
                print(f"  - {r}")
            print("\nPer Phase 4.2 specifications, full training cannot proceed on CPU / offline fallback.")
            print("Please mount the Google Colab environment with NVIDIA Tesla T4 GPU to launch.")
            return 1

        print("🚀 Launching Production Supervised Fine-Tuning...")
        trainer = ProductionSFTTrainer(config)
        telemetry = trainer.train(
            resume_from_checkpoint=resume_ckpt,
            max_steps=args.max_steps,
            override_epochs=args.epochs,
        )
        print("\n🎉 Training Complete!")
        print(telemetry.to_markdown())

    return 0


if __name__ == "__main__":
    sys.exit(main())
