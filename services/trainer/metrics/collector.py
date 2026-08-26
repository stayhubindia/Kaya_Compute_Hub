from typing import Dict, Any, List, Optional
from apps.training.models import TrainingRun, TrainingMetric
from services.trainer.metrics.schemas import MetricRecord, MAX_METRICS_PER_STEP

class MetricCollector:
    def __init__(self, training_run_id: str, target_metric_name: str = "val_loss", metric_direction: str = "minimize"):
        self.training_run_id = str(training_run_id)
        self.target_metric_name = target_metric_name
        self.metric_direction = metric_direction
        self.best_metric_val: Optional[float] = None

    def record_metrics(self, records: List[MetricRecord]):
        if len(records) > MAX_METRICS_PER_STEP:
            records = records[:MAX_METRICS_PER_STEP]

        metric_objs = []
        for rec in records:
            rec.validate()
            metric_objs.append(
                TrainingMetric(
                    training_run_id=self.training_run_id,
                    step=rec.step,
                    epoch=rec.epoch,
                    name=rec.name,
                    value=rec.value,
                    split=rec.split
                )
            )

        TrainingMetric.objects.bulk_create(metric_objs)

        # Update best metric & current step/epoch on TrainingRun
        if records:
            last_rec = records[-1]
            run_updates: Dict[str, Any] = {
                "current_epoch": last_rec.epoch,
                "current_step": last_rec.step,
            }

            for rec in records:
                if rec.name == self.target_metric_name:
                    if self.best_metric_val is None:
                        self.best_metric_val = rec.value
                        run_updates["best_metric"] = rec.value
                        run_updates["best_metric_name"] = rec.name
                    elif self.metric_direction == "minimize" and rec.value < self.best_metric_val:
                        self.best_metric_val = rec.value
                        run_updates["best_metric"] = rec.value
                        run_updates["best_metric_name"] = rec.name
                    elif self.metric_direction == "maximize" and rec.value > self.best_metric_val:
                        self.best_metric_val = rec.value
                        run_updates["best_metric"] = rec.value
                        run_updates["best_metric_name"] = rec.name

            TrainingRun.objects.filter(id=self.training_run_id).update(**run_updates)
