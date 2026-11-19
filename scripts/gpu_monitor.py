"""Three-second device utilization, memory and power samples."""
import subprocess
import threading
import time
from experiment_io import append

def monitor_gpu(out, attempt, start):
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                    '--format=csv,noheader,nounits'], text=True).strip()
                append(out/'gpu_samples.jsonl', dict(attempt=attempt, seconds=time.perf_counter()-start, gpu=value))
            except Exception: pass
            stop.wait(3)
    thread = threading.Thread(target=sample, daemon=True); thread.start()
    return stop, thread
