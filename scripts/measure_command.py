"""Measure a command without requiring an extra system package."""
import argparse
import json
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    start = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    stamp = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, check=False)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = dict(command=command, started_at_utc=stamp, exit_code=result.returncode,
                  wall_seconds=time.perf_counter()-start,
                  child_user_cpu_seconds=after.ru_utime-before.ru_utime,
                  child_system_cpu_seconds=after.ru_stime-before.ru_stime,
                  maximum_child_peak_rss_kib=after.ru_maxrss,
                  gpu_compute_used=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
