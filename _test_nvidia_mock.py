"""Mock-pynvml verification of the NVIDIA collector path.

We can't run real NVIDIA on this Mac, but we can inject a fake `pynvml`
module via sys.modules BEFORE trackio_html starts the system monitor.
The mock simulates 2 GPUs with realistic varying values, and we verify:
  - GPU info is captured into the run config
  - system/gpu.<i>.* metrics are written
  - per-process memory uses *this* process's PID
  - both GPUs produce independent series
"""

import json
import os
import re
import sys
import time
import types


def make_fake_pynvml():
    m = types.ModuleType("pynvml")
    m.NVML_TEMPERATURE_GPU = 0
    state = {"tick": 0}

    def nvmlInit():
        pass

    def nvmlShutdown():
        pass

    def nvmlDeviceGetCount():
        return 2

    def nvmlDeviceGetHandleByIndex(i):
        return ("handle", i)

    def nvmlDeviceGetName(h):
        return f"NVIDIA Tesla TEST-{h[1]}".encode()

    class _Mem:
        def __init__(self, total, used):
            self.total = total
            self.used = used
            self.free = total - used

    def nvmlDeviceGetMemoryInfo(h):
        idx = h[1]
        total = (24 if idx == 0 else 16) * 1024 * 1024 * 1024
        base = (4 + idx * 2) * 1024 * 1024 * 1024
        used = base + state["tick"] * 200 * 1024 * 1024
        used = min(used, total - 1024 * 1024 * 1024)
        return _Mem(total, used)

    class _Util:
        def __init__(self, gpu, memory):
            self.gpu = gpu
            self.memory = memory

    def nvmlDeviceGetUtilizationRates(h):
        idx = h[1]
        gpu = (40 + idx * 10 + state["tick"] * 5) % 100
        mem = (20 + idx * 5 + state["tick"] * 3) % 100
        return _Util(gpu, mem)

    def nvmlDeviceGetPowerUsage(h):
        return 180_000 + h[1] * 30_000 + state["tick"] * 5_000

    def nvmlDeviceGetTemperature(h, kind):
        return 55 + h[1] * 3 + state["tick"]

    class _Proc:
        def __init__(self, pid, usedGpuMemory):
            self.pid = pid
            self.usedGpuMemory = usedGpuMemory

    def nvmlDeviceGetComputeRunningProcesses_v3(h):
        my = os.getpid()
        return [
            _Proc(pid=99999, usedGpuMemory=512 * 1024 * 1024),
            _Proc(pid=my, usedGpuMemory=(2 + h[1]) * 1024 * 1024 * 1024),
        ]

    def _tick():
        state["tick"] += 1

    m.nvmlInit = nvmlInit
    m.nvmlShutdown = nvmlShutdown
    m.nvmlDeviceGetCount = nvmlDeviceGetCount
    m.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    m.nvmlDeviceGetName = nvmlDeviceGetName
    m.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo
    m.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates
    m.nvmlDeviceGetPowerUsage = nvmlDeviceGetPowerUsage
    m.nvmlDeviceGetTemperature = nvmlDeviceGetTemperature
    m.nvmlDeviceGetComputeRunningProcesses_v3 = nvmlDeviceGetComputeRunningProcesses_v3
    m._tick = _tick
    return m


fake = make_fake_pynvml()
sys.modules["pynvml"] = fake

import trackio_html as wandb

OUT = "trackio_html"
PROJ = "nvidia-mock"
for ext in ("jsonl", "html"):
    p = f"{OUT}/{PROJ}.{ext}"
    if os.path.exists(p):
        os.remove(p)

wandb.init(
    project=PROJ,
    name="mock-2gpu",
    config={"lr": 1e-3, "bs": 32},
    system_interval=0.5,
)

for step in range(20):
    wandb.log({"train/loss": 1.0 / (step + 1), "train/acc": min(1.0, step / 20)})
    fake._tick()
    time.sleep(0.1)

wandb.finish()

html = open(f"{OUT}/{PROJ}.html").read()
m = re.search(r'<script id="trackio-data"[^>]*>(.*?)</script>', html, re.S)
data = json.loads(m.group(1).replace("<\\/", "</"))
run = next(iter(data["runs"].values()))

print("== run name:", run["name"])
print("== config (gpu fields):")
for k, v in run["config"].items():
    if k.startswith("gpu."):
        print(f"   {k}: {v}")

print("\n== system/gpu.* metrics:")
sys_keys = sorted(k for k in run["metrics"] if k.startswith("system/gpu."))
for k in sys_keys:
    pts = run["metrics"][k]
    vals = [round(p[1], 2) for p in pts]
    print(f"   {k}: {len(pts)} points → {vals}")

print("\n== checks:")
checks = {
    "gpu.0 in config": "gpu.0" in run["config"],
    "gpu.1 in config": "gpu.1" in run["config"],
    "gpu.0.util series exists": "system/gpu.0.util" in run["metrics"],
    "gpu.1.util series exists": "system/gpu.1.util" in run["metrics"],
    "gpu.0.proc_mem_mb captured": "system/gpu.0.proc_mem_mb" in run["metrics"],
    "gpu.1.proc_mem_mb captured": "system/gpu.1.proc_mem_mb" in run["metrics"],
    "gpu.0.temp_c captured": "system/gpu.0.temp_c" in run["metrics"],
    "gpu.0.power_w captured": "system/gpu.0.power_w" in run["metrics"],
    "proc_mem_mb gpu.0 == 2048": (
        run["metrics"].get("system/gpu.0.proc_mem_mb", [[0, 0]])[0][1] == 2048.0
    ),
    "proc_mem_mb gpu.1 == 3072": (
        run["metrics"].get("system/gpu.1.proc_mem_mb", [[0, 0]])[0][1] == 3072.0
    ),
}
all_ok = True
for k, v in checks.items():
    print(f"   [{'OK' if v else 'FAIL'}] {k}")
    if not v:
        all_ok = False

sys.exit(0 if all_ok else 1)
