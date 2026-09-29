"""Manually queue iteration 5 after the current iteration-4 process completes.

Run under caffeinate. --dry-run only inspects readiness and prints the command.
"""
import argparse
import datetime
import fcntl
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from challenge_lib import paths

PREVIOUS = paths.experiment_results("iteration_4_stable_deterministic") / "lat32_ep500_ntr4500"
TARGET = paths.experiment_results("iteration_5_higher_lr") / "lat32_ep500_ntr4500"
QUEUE_DIR = paths.ARTIFACTS_DIR / "queues"


def process_info(pid):
    result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                            capture_output=True, text=True)
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return None
    if result.returncode != 0:
        raise RuntimeError(f"Cannot inspect predecessor process: {result.stderr.strip()}")
    return result.stdout.strip() or None


def check_process(info):
    if info is None:
        return
    fields = info.split(None, 5)
    if len(fields) != 6:
        raise ValueError("Unexpected process identity")
    args = shlex.split(fields[5])
    if not args or "python" not in Path(args[0]).name.lower() or not any(
        token.endswith("iteration_4_stable_deterministic/train_stable.py") for token in args[1:]
    ):
        raise ValueError("PID does not identify the iteration-4 Python training process")


def completed_successfully(folder):
    p = json.loads((folder / "parameters.json").read_text())
    if not p.get("finished") or p.get("epochs") != 500 or p.get("n_train") != 4500 or p.get("n_val") != 500:
        raise RuntimeError("Iteration 4 has not recorded successful completion; iteration 5 will not start")
    histories = list(folder.glob("history_*.json"))
    metrics = list(folder.glob("metrics_*.json"))
    if len(histories) != 1 or len(metrics) != 1:
        raise RuntimeError("Expected one completed history and metrics file for iteration 4")
    h, m = json.loads(histories[0].read_text()), json.loads(metrics[0].read_text())
    if len(h["tr_loss"]) != 500 or h["va_epoch"][-1] != 500 or not math.isfinite(m["rootNMSE"]):
        raise RuntimeError("Iteration 4 completion files are incomplete or non-finite")


def check_target():
    if (TARGET / "parameters.json").exists() or any(TARGET.glob("*.pt")):
        raise FileExistsError(f"Iteration 5 already has a run in {TARGET}; refusing duplicate launch")


def command():
    return [sys.executable, "-u", str(ROOT / "iteration_5_higher_lr/train_higher_lr.py"),
            "--epochs", "500", "--latent", "32", "--lr", "0.0003", "--min-lr", "0.00003",
            "--grad-clip", "1.0", "--val-every", "5", "--batch-size", "32", "--device", "mps"]


def dispatch(pid, poll_seconds=30, dry_run=False):
    check_target()
    identity = process_info(pid)
    check_process(identity)
    if identity is None:
        completed_successfully(PREVIOUS)
    print("Will run:", shlex.join(command()), flush=True)
    print(f"Predecessor PID {pid}: {'still running' if identity else 'finished successfully'}", flush=True)
    if dry_run:
        print("Dry run only: no waiting, process launch or output writes.")
        return 0

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with (QUEUE_DIR / "iteration5.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another iteration-5 queue is already active") from exc
        status_path = QUEUE_DIR / "iteration5.json"

        def status(state, **extra):
            record = {"state": state, "after_pid": pid, "predecessor_identity": identity,
                      "updated": datetime.datetime.now().isoformat(), "command": command(), **extra}
            temp = status_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(record, indent=2) + "\n")
            temp.replace(status_path)

        try:
            status("waiting")
            while identity is not None:
                current = process_info(pid)
                if current is None:
                    break
                if current != identity:
                    raise RuntimeError("Predecessor PID was reused or its identity changed; refusing automatic launch")
                time.sleep(poll_seconds)
            completed_successfully(PREVIOUS)
            check_target()
            status("running")
            print("Iteration 4 completed successfully. Starting iteration 5 now.", flush=True)
            result = subprocess.run(command(), cwd=ROOT)
            status("completed" if result.returncode == 0 else "failed", returncode=result.returncode)
            return result.returncode
        except (Exception, KeyboardInterrupt) as exc:
            status("cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--after-pid", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.after_pid <= 1:
        p.error("after-pid must identify the running Python process")
    return dispatch(args.after_pid, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
