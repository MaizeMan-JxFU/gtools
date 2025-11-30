import psutil
import os
def get_process_info():
    """Usage of CPU and memory"""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024**3  # GB
    return cpu_percent, memory_mb