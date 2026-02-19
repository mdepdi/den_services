import time

def report_state(task, *, state, message="", current=None, total=None, extra=None):
    """
    Standard Celery progress meta schema.
    - percent is 0..100
    - adds timestamp for debugging
    """
    meta = {
        "state": state,
        "status": str(message),
        "ts": time.time(),
    }

    if current is not None and total is not None and total > 0:
        meta["progress"] = {
            "current": int(current),
            "total": int(total),
            "percent": round((current * 100) / total, 2),
        }

    if extra:
        meta.update(extra)

    if task:
        task.update_state(state=state, meta=meta)
