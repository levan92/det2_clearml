import time
import psutil
import torch.multiprocessing as mp


def monitor(freq=1):
    while True:
        time.sleep(freq)
        print(psutil.disk_usage("/dev/shm"))
        print(psutil.virtual_memory())


def start_monitor(freq=1):
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    p = mp.Process(target=monitor, daemon=True, args=(freq,))
    p.start()
