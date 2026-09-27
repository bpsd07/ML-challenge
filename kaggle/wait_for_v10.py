import subprocess
import time
import sys

KERNEL_ID = "bhanupratapsinghdeo/ml-challenge-blocking-optimization"

print(f"Monitoring Kaggle kernel: {KERNEL_ID}", flush=True)
while True:
    try:
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", KERNEL_ID],
            capture_output=True,
            text=True,
            timeout=30
        )
        out = res.stdout.strip()
        print(f"[{time.strftime('%H:%M:%S')}] {out}", flush=True)
        
        if "KernelWorkerStatus.RUNNING" not in out and "RUNNING" not in out:
            print(f"Kernel execution completed with output: {out}", flush=True)
            break
    except Exception as e:
        print(f"Error checking status: {e}", flush=True)
        
    time.sleep(45)
