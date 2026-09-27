"""Launch local training independently of the chat and record process completion."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-name", default="submission5_local")
    args = parser.parse_args()
    launch = ROOT / "models" / f"{args.run_name}_launch"
    if args.worker:
        command = ["/usr/bin/caffeinate", "-i", sys.executable, "-u",
                   str(ROOT / "src/train_submission5.py"), "--device", "mps",
                   "--run-name", args.run_name,
                   "--pretrained", str(ROOT / "models/submission5_benchmark/pretrained")]
        started = timestamp()
        try:
            result = subprocess.run(command, cwd=ROOT, check=False)
            status = {"started_utc": started, "finished_utc": timestamp(), "exit_code": result.returncode}
        except Exception as error:
            status = {"started_utc": started, "finished_utc": timestamp(), "error": repr(error)}
        (launch / "completion.json").write_text(json.dumps(status, indent=2) + "\n")
        return
    if (ROOT / "models" / args.run_name).exists():
        raise FileExistsError(f"Preserving existing {args.run_name} run")
    launch.mkdir()  # Exclusive launch: never start two jobs or overwrite logs.
    with (launch / "training.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--worker", "--run-name", args.run_name],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    status = {"worker_pid": process.pid, "started_utc": timestamp(),
              "run_name": args.run_name, "log": str(launch / "training.log"),
              "completion": str(launch / "completion.json")}
    (launch / "launch.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
