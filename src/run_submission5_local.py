"""Launch local training independently of the chat and record process completion."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "models/submission5_launch"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        command = ["/usr/bin/caffeinate", "-i", sys.executable, "-u",
                   str(ROOT / "src/train_submission5.py"), "--device", "mps",
                   "--pretrained", str(ROOT / "models/submission5_benchmark/pretrained")]
        started = timestamp()
        try:
            result = subprocess.run(command, cwd=ROOT, check=False)
            status = {"started_utc": started, "finished_utc": timestamp(), "exit_code": result.returncode}
        except Exception as error:
            status = {"started_utc": started, "finished_utc": timestamp(), "error": repr(error)}
        (LAUNCH / "completion.json").write_text(json.dumps(status, indent=2) + "\n")
        return
    if (ROOT / "models/submission5").exists():
        raise FileExistsError("Preserving existing submission5 run")
    LAUNCH.mkdir()  # Exclusive launch: never start two jobs or overwrite logs.
    with (LAUNCH / "training.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    status = {"worker_pid": process.pid, "started_utc": timestamp(),
              "log": str(LAUNCH / "training.log"), "completion": str(LAUNCH / "completion.json")}
    (LAUNCH / "launch.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
