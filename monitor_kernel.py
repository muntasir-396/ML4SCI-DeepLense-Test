import time
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = "muntasirmamun396/ml4sci-deeplense-common-test-1"
print(f"Monitoring Kaggle kernel: {slug}")

start_time = time.time()
while True:
    try:
        res = api.kernels_status(slug)
        d = res.to_dict()
        status = str(d.get("status", "")).upper()
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed:03d}s] Kernel status: {status}", flush=True)
        
        if status in ["COMPLETE", "ERROR", "CANCELLED"]:
            print(f"Final status reached: {status}", flush=True)
            if d.get("failureMessage") or d.get("failure_message"):
                msg = d.get("failureMessage") or d.get("failure_message")
                print(f"Failure message: {msg}", flush=True)
            break
    except Exception as e:
        print(f"Exception during check: {e}", flush=True)
        
    time.sleep(20)
