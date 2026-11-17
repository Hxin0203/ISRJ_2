"""Server-only entry point for the independent normalization factor."""
from .train import main

if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import os
        import time
        import traceback
        from paper2.io import now
        from tools.durable_io import atomic_json
        from .train import BASE
        from .runtime import require_server
        # Do not write run artifacts on a workstation if the host guard fails.
        require_server()
        failure = {"stage": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
                   "pid": os.getpid(), "updated_utc": now()}
        atomic_json(BASE / f"CONTROLLER_FAILURE_{time.time_ns()}.json", failure)
        atomic_json(BASE / "STATUS.json", failure)
        raise
