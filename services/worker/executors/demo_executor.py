import time
from typing import Callable
from services.worker.executors.pipeline_executor import PIPELINE_EXECUTORS, run_pipeline_executor

def simulate_download(job_payload: dict, update_progress_cb: Callable[[int, str, str], None]) -> dict:
    update_progress_cb(10, "downloading", "Initializing stream connection")
    time.sleep(0.05)
    update_progress_cb(50, "downloading", "Transferring data chunks (simulated)")
    time.sleep(0.05)
    update_progress_cb(100, "completed", "Download complete")
    return {"bytes_downloaded": 1048576, "status": "simulated_success"}

def simulate_extraction(job_payload: dict, update_progress_cb: Callable[[int, str, str], None]) -> dict:
    update_progress_cb(20, "extracting", "Reading archive headers")
    time.sleep(0.05)
    update_progress_cb(70, "extracting", "Uncompressing archive contents")
    time.sleep(0.05)
    update_progress_cb(100, "completed", "Archive extraction complete")
    return {"files_extracted": 42, "status": "simulated_success"}

def simulate_preprocessing(job_payload: dict, update_progress_cb: Callable[[int, str, str], None]) -> dict:
    update_progress_cb(15, "preprocessing", "Parsing data structures")
    time.sleep(0.05)
    update_progress_cb(60, "preprocessing", "Cleaning dataset entries")
    time.sleep(0.05)
    update_progress_cb(100, "completed", "Dataset preprocessing complete")
    return {"records_processed": 1000, "status": "simulated_success"}

APPROVED_DEMO_EXECUTORS = {
    'download': simulate_download,
    'extraction': simulate_extraction,
    'preprocessing': simulate_preprocessing,
}

def run_approved_executor(job_type: str, job_payload: dict, update_progress_cb: Callable[[int, str, str], None]) -> dict:
    """
    Executes ONLY approved handlers from a strict allowlist.
    Rejects any unapproved job types, shell execution, or code injection.
    """
    if job_type in PIPELINE_EXECUTORS:
        return run_pipeline_executor(job_type, job_payload, update_progress_cb)

    executor_fn = APPROVED_DEMO_EXECUTORS.get(job_type)
    if not executor_fn:
        raise NotImplementedError(f"Job type '{job_type}' is not supported by the worker environment.")
    return executor_fn(job_payload, update_progress_cb)
