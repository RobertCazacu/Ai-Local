import json
import os
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Dict


def _jobs_path(store_dir: str) -> str:
    return os.path.join(store_dir, "jobs.jsonl")


def _logs_dir(store_dir: str) -> str:
    d = os.path.join(store_dir, "logs")
    os.makedirs(d, exist_ok=True)
    return d


def append_job(store_dir: str, payload: Dict) -> None:
    with open(_jobs_path(store_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


@contextmanager
def job_context(store_dir: str, job_type: str):
    job_id = uuid.uuid4().hex[:10]
    started = datetime.utcnow().isoformat()
    log_path = os.path.join(_logs_dir(store_dir), f"job_{job_id}.log")
    meta = {"id": job_id, "type": job_type, "status": "running", "start": started, "end": None, "log_path": log_path}
    append_job(store_dir, meta)
    try:
        with open(log_path, "a", encoding="utf-8") as logf:
            def log(msg: str):
                logf.write(f"[{datetime.utcnow().isoformat()}] {msg}\n")
                logf.flush()
            yield job_id, log
        meta["status"] = "done"
    except Exception:
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(traceback.format_exc() + "\n")
        meta["status"] = "failed"
        raise
    finally:
        meta["end"] = datetime.utcnow().isoformat()
        append_job(store_dir, meta)
