"""Local schedule helper: collect a finished run before optionally shutting down AutoDL."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HOST = "autodl-bigquant"
REMOTE = "/root/autodl-tmp/minute-grid-240-20260906/results"
LOCAL = (
    Path(__file__).resolve().parents[3] / "reports/dependencies/finals_pre/minute_grid_240/20260906"
)


def ssh(command):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, command],
        text=True,
        timeout=60,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    status = json.loads(ssh("cat " + REMOTE + "/status.json"))
    if status["state"] not in ("completed", "failed"):
        # A dead supervisor is an exception, not an endlessly running experiment.
        try:
            ssh("kill -0 " + str(int(status["pid"])))
        except subprocess.CalledProcessError:
            raise RuntimeError("supervisor exited without a final status; inspect runner.log")
        print(json.dumps(status, ensure_ascii=False))
        return 0
    LOCAL.mkdir(parents=True, exist_ok=True)
    checks = ssh("cd " + REMOTE + " && find . -type f -print0 | sort -z | xargs -0 sha256sum")
    subprocess.run(["scp", "-r", HOST + ":" + REMOTE + "/.", str(LOCAL)], check=True, timeout=180)
    count = 0
    for line in checks.splitlines():
        expected, relative = line.split("  ", 1)
        path = LOCAL / relative
        if not path.resolve().is_relative_to(LOCAL.resolve()):
            raise RuntimeError("unexpected artifact path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("artifact checksum mismatch: " + relative)
        count += 1
    receipt = {
        "remote": HOST + ":" + REMOTE,
        "state": status["state"],
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "verified_files": count,
    }
    (LOCAL / "collection_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.shutdown:
        busy = ssh("nvidia-smi --query-compute-apps=pid --format=csv,noheader").strip()
        if busy:
            raise RuntimeError("GPU still has active processes; inspect before shutdown: " + busy)
        receipt["shutdown_requested_at"] = datetime.now(timezone.utc).isoformat()
        # This provider command stops the container immediately. SSH may disconnect.
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, "shutdown"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,  # Preserve SSH's disconnect/exit status in the shutdown receipt.
        )
        receipt["shutdown_ssh_returncode"] = result.returncode
        receipt["shutdown_output"] = result.stdout + result.stderr
        (LOCAL / "collection_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
