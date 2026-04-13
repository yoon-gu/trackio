"""trackio_html — wandb-compatible tracker that writes a self-contained HTML dashboard.

Usage:
    import trackio_html as wandb

    wandb.init(project="credit-model", name="exp-1", config={"lr": 0.01})
    for step in range(100):
        wandb.log({"loss": loss, "acc": acc})
    wandb.finish()

Outputs (relative to CWD by default):
    trackio_html/<project>.jsonl   append-only event log (source of truth)
    trackio_html/<project>.html    self-contained dashboard, no server needed

All runs sharing the same project accumulate in the same HTML and can be
compared side-by-side. The HTML has no external dependencies — all charts
are rendered as inline SVG, suitable for locked-down environments.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["init", "log", "finish", "config", "Run"]

_STATE: Dict[str, Any] = {"run": None}


def _to_scalar(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if hasattr(v, "item"):
        try:
            return _to_scalar(v.item())
        except Exception:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        try:
            return o.tolist()
        except Exception:
            pass
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


class _ConfigProxy:
    """Attribute/dict access to the active run's config, mimicking wandb.config."""

    def _target(self) -> Dict[str, Any]:
        run = _STATE["run"]
        return {} if run is None else run.config

    def __setattr__(self, key: str, value: Any) -> None:
        run = _STATE["run"]
        if run is not None:
            run.config[key] = value

    def __getattr__(self, key: str) -> Any:
        t = self._target()
        if key in t:
            return t[key]
        raise AttributeError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        run = _STATE["run"]
        if run is not None:
            run.config[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._target()[key]

    def update(self, d: Dict[str, Any]) -> None:
        run = _STATE["run"]
        if run is not None:
            run.config.update(d)

    def __repr__(self) -> str:
        return f"Config({self._target()!r})"


config = _ConfigProxy()


class _SystemMonitor:
    """Background sampler for GPU / CPU / RAM. Logs under `system/` group.

    Soft dependencies — silently skips collectors whose libs aren't importable:
      - pynvml      → NVIDIA GPU util/mem/power/temp (one set of metrics per GPU)
      - psutil      → CPU%, RAM% / used MB
      - torch.mps   → Apple Silicon allocated GPU memory

    The sampler runs in a daemon thread and uses *elapsed seconds since run
    start* as the step value, so system metrics live on their own time axis
    (independent from the user's training step counter).
    """

    def __init__(self, run: "Run", interval: float) -> None:
        self.run = run
        self.interval = max(0.5, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = time.time()
        self._pid = os.getpid()
        self._pynvml = None
        self._gpu_handles: list = []
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)
            ]
        except Exception:
            self._pynvml = None
            self._gpu_handles = []
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None
        self._torch_mps = None
        try:
            import torch  # type: ignore

            if hasattr(torch, "backends") and torch.backends.mps.is_available():
                self._torch_mps = torch
        except Exception:
            self._torch_mps = None

    def has_collectors(self) -> bool:
        return bool(self._gpu_handles or self._psutil or self._torch_mps)

    def gpu_info(self) -> Dict[str, str]:
        info: Dict[str, str] = {}
        if not (self._pynvml and self._gpu_handles):
            return info
        p = self._pynvml
        for i, h in enumerate(self._gpu_handles):
            try:
                name = p.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "ignore")
                mem = p.nvmlDeviceGetMemoryInfo(h)
                info[f"gpu.{i}"] = f"{name} ({mem.total / (1024**3):.1f} GB)"
            except Exception:
                continue
        return info

    def _get_proc_list(self, h):
        p = self._pynvml
        for fn_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            fn = getattr(p, fn_name, None)
            if fn is None:
                continue
            try:
                return fn(h)
            except Exception:
                continue
        return []

    def start(self) -> None:
        if not self.has_collectors():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            sample = self._sample()
            if not sample:
                continue
            elapsed = max(1, int(time.time() - self._t0))
            try:
                self.run._log_system(sample, step=elapsed)
            except Exception:
                pass

    def _sample(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self._gpu_handles and self._pynvml is not None:
            p = self._pynvml
            for i, h in enumerate(self._gpu_handles):
                try:
                    util = p.nvmlDeviceGetUtilizationRates(h)
                    mem = p.nvmlDeviceGetMemoryInfo(h)
                    prefix = f"system/gpu.{i}"
                    out[f"{prefix}.util"] = float(util.gpu)
                    out[f"{prefix}.mem_util"] = float(util.memory)
                    out[f"{prefix}.mem_used_mb"] = mem.used / (1024 * 1024)
                    out[f"{prefix}.mem_pct"] = mem.used / mem.total * 100.0
                    try:
                        out[f"{prefix}.power_w"] = p.nvmlDeviceGetPowerUsage(h) / 1000.0
                    except Exception:
                        pass
                    try:
                        out[f"{prefix}.temp_c"] = float(
                            p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU)
                        )
                    except Exception:
                        pass
                    try:
                        for proc in self._get_proc_list(h):
                            if getattr(proc, "pid", None) == self._pid:
                                used = getattr(proc, "usedGpuMemory", None)
                                if used is not None:
                                    out[f"{prefix}.proc_mem_mb"] = float(used) / (
                                        1024 * 1024
                                    )
                                break
                    except Exception:
                        pass
                except Exception:
                    continue
        if self._psutil is not None:
            try:
                out["system/cpu.util"] = float(self._psutil.cpu_percent(interval=None))
                vm = self._psutil.virtual_memory()
                out["system/cpu.mem_pct"] = float(vm.percent)
                out["system/cpu.mem_used_mb"] = (vm.total - vm.available) / (
                    1024 * 1024
                )
            except Exception:
                pass
        if self._torch_mps is not None:
            try:
                t = self._torch_mps
                out["system/mps.mem_alloc_mb"] = t.mps.current_allocated_memory() / (
                    1024 * 1024
                )
                if hasattr(t.mps, "driver_allocated_memory"):
                    out["system/mps.mem_driver_mb"] = (
                        t.mps.driver_allocated_memory() / (1024 * 1024)
                    )
            except Exception:
                pass
        return out


class Run:
    def __init__(
        self,
        project: str,
        name: Optional[str],
        config: Optional[Dict[str, Any]],
        output_dir: Path,
        render_interval: float = 2.0,
        monitor_system: bool = True,
        system_interval: float = 5.0,
    ) -> None:
        self.project = project
        self.name = name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.id = f"{self.name}-{int(time.time() * 1000)}"
        self.config: Dict[str, Any] = dict(config or {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / f"{project}.jsonl"
        self.html_path = self.output_dir / f"{project}.html"
        self._step = 0
        self._start_time = time.time()
        self._last_render = 0.0
        self._render_interval = render_interval
        self._finished = False
        self._write_lock = threading.Lock()
        self._sysmon: Optional[_SystemMonitor] = None
        if monitor_system:
            mon = _SystemMonitor(self, interval=system_interval)
            if mon.has_collectors():
                self._sysmon = mon
                for k, v in mon.gpu_info().items():
                    self.config.setdefault(k, v)
        self._append_event(
            {
                "type": "run_start",
                "run_id": self.id,
                "name": self.name,
                "config": dict(self.config),
                "timestamp": self._start_time,
            }
        )
        self._render()
        if self._sysmon is not None:
            self._sysmon.start()

    def _append_event(self, ev: Dict[str, Any]) -> None:
        line = json.dumps(ev, default=_json_default, ensure_ascii=False) + "\n"
        with self._write_lock:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def _log_system(self, sample: Dict[str, float], step: int) -> None:
        if self._finished or not sample:
            return
        self._append_event(
            {
                "type": "log",
                "run_id": self.id,
                "step": step,
                "timestamp": time.time(),
                "data": sample,
            }
        )

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._finished:
            raise RuntimeError("trackio_html: log() called on a finished run.")
        if step is None:
            step = self._step
        self._step = step + 1
        cleaned: Dict[str, float] = {}
        for k, v in data.items():
            s = _to_scalar(v)
            if s is None:
                continue
            cleaned[str(k)] = s
        if not cleaned:
            return
        self._append_event(
            {
                "type": "log",
                "run_id": self.id,
                "step": step,
                "timestamp": time.time(),
                "data": cleaned,
            }
        )
        now = time.time()
        if now - self._last_render >= self._render_interval:
            self._render()
            self._last_render = now

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._sysmon is not None:
            self._sysmon.stop()
            self._sysmon = None
        self._append_event(
            {"type": "run_end", "run_id": self.id, "timestamp": time.time()}
        )
        self._render()

    def _render(self) -> None:
        runs_data = _parse_jsonl(self.jsonl_path)
        _write_html(self.project, runs_data, self.html_path)

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()


def _parse_jsonl(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    runs: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return runs
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = ev.get("run_id")
                if not rid:
                    continue
                r = runs.setdefault(
                    rid,
                    {
                        "name": rid,
                        "config": {},
                        "metrics": {},
                        "start_time": None,
                        "end_time": None,
                    },
                )
                t = ev.get("type")
                if t == "run_start":
                    r["name"] = ev.get("name", rid)
                    r["config"] = ev.get("config") or {}
                    r["start_time"] = ev.get("timestamp")
                elif t == "log":
                    step = ev.get("step", 0)
                    for k, v in (ev.get("data") or {}).items():
                        if isinstance(v, (int, float)):
                            r["metrics"].setdefault(k, []).append([step, v])
                elif t == "run_end":
                    r["end_time"] = ev.get("timestamp")
    except OSError:
        pass
    return runs


def init(
    project: str = "default",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    dir: str = "./trackio_html",
    render_interval: float = 2.0,
    monitor_system: bool = True,
    system_interval: float = 5.0,
    **_: Any,
) -> Run:
    run = Run(
        project=project,
        name=name,
        config=config,
        output_dir=Path(dir),
        render_interval=render_interval,
        monitor_system=monitor_system,
        system_interval=system_interval,
    )
    _STATE["run"] = run
    return run


def log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    run = _STATE["run"]
    if run is None:
        raise RuntimeError("trackio_html: call init() before log().")
    run.log(data, step=step)


def finish() -> None:
    run = _STATE["run"]
    if run is None:
        return
    run.finish()
    _STATE["run"] = None


def __getattr__(name: str) -> Any:
    if name == "run":
        return _STATE["run"]
    raise AttributeError(name)


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _write_html(
    project: str, runs_data: Dict[str, Dict[str, Any]], html_path: Path
) -> None:
    payload = {
        "project": project,
        "runs": runs_data,
        "generated_at": time.time(),
    }
    data_json = json.dumps(payload, default=_json_default, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")
    html = _HTML_TEMPLATE.replace("__TRACKIO_DATA_PLACEHOLDER__", data_json)
    html = html.replace("__TRACKIO_PROJECT_PLACEHOLDER__", _html_escape(project))
    tmp = html_path.with_suffix(html_path.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, html_path)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>trackio · __TRACKIO_PROJECT_PLACEHOLDER__</title>
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; margin: 0; padding: 24px; background: #f7f8fa; color: #111827; }
header { display: flex; align-items: baseline; gap: 14px; margin-bottom: 18px; }
header h1 { margin: 0; font-size: 20px; font-weight: 600; }
header .subtitle { color: #6b7280; font-size: 13px; }
header .generated { color: #9ca3af; font-size: 11px; margin-left: auto; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.layout { display: grid; grid-template-columns: 300px 1fr; gap: 20px; align-items: start; }
.panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }
.runs-panel { max-height: calc(100vh - 110px); overflow-y: auto; position: sticky; top: 20px; }
.runs-panel h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; margin: 0 0 12px; }
.runs-panel .bulk { font-size: 11px; color: #2563eb; cursor: pointer; user-select: none; margin-bottom: 10px; display: inline-block; }
.runs-panel .bulk:hover { text-decoration: underline; }
.run-item { padding: 10px 0; border-top: 1px solid #f1f3f5; }
.run-item:first-of-type { border-top: none; padding-top: 0; }
.run-head { display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; }
.run-head input { margin: 0; }
.run-swatch { width: 10px; height: 10px; border-radius: 2px; flex: 0 0 10px; }
.run-name { font-weight: 500; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.run-meta { font-size: 10px; color: #9ca3af; margin: 4px 0 0 26px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.run-config { margin: 6px 0 0 26px; font-size: 11px; color: #6b7280; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.55; }
.run-config span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.metric-group { margin-bottom: 18px; }
.metric-group-header { display: flex; align-items: center; gap: 10px; margin: 0 0 10px; cursor: pointer; user-select: none; }
.metric-group-header h2 { margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; font-weight: 600; }
.metric-group-header .count { font-size: 10px; color: #9ca3af; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.metric-group-header .caret { font-size: 10px; color: #9ca3af; transition: transform 0.15s; }
.metric-group.collapsed .caret { transform: rotate(-90deg); }
.metric-group.collapsed .charts-grid { display: none; }
.charts-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; }
.chart-container { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 14px 10px; }
.chart-container h3 { margin: 0 0 6px; font-size: 13px; font-weight: 600; color: #111827; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; display: flex; align-items: center; gap: 8px; }
.chart-container h3 .key { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chart-container h3 .log-toggle { font-size: 10px; color: #6b7280; background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 4px; padding: 2px 6px; cursor: pointer; font-family: inherit; }
.chart-container h3 .log-toggle.active { background: #2563eb; color: #fff; border-color: #2563eb; }
.chart { width: 100%; display: block; }
.chart .grid { stroke: #f1f3f5; stroke-width: 1; }
.chart .axis { stroke: #d1d5db; stroke-width: 1; }
.chart .tick { font-size: 10px; fill: #6b7280; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.chart .cursor { stroke: #9ca3af; stroke-width: 1; stroke-dasharray: 3 3; display: none; }
.tooltip { position: fixed; background: rgba(17, 24, 39, 0.96); color: #fff; padding: 8px 10px; border-radius: 6px; font-size: 11px; pointer-events: none; z-index: 1000; display: none; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; box-shadow: 0 4px 14px rgba(0,0,0,0.18); max-width: 260px; }
.tooltip .ttrow { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
.tooltip .ttrow .sw { width: 8px; height: 8px; border-radius: 1px; flex: 0 0 8px; }
.tooltip .ttrow .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tooltip .ttstep { font-weight: 600; color: #e5e7eb; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 4px; margin-bottom: 2px; }
.empty { color: #6b7280; padding: 40px; text-align: center; font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>__TRACKIO_PROJECT_PLACEHOLDER__</h1>
  <span class="subtitle" id="subtitle"></span>
  <span class="generated" id="generated"></span>
</header>
<div class="layout">
  <aside class="panel runs-panel">
    <h2>Runs</h2>
    <span class="bulk" id="bulk-toggle">toggle all</span>
    <div id="runs-list"></div>
  </aside>
  <main>
    <div id="charts"></div>
  </main>
</div>
<div class="tooltip" id="tooltip"></div>
<script id="trackio-data" type="application/json">__TRACKIO_DATA_PLACEHOLDER__</script>
<script>
(function () {
  var raw = document.getElementById('trackio-data').textContent;
  var DATA = JSON.parse(raw);
  var runs = DATA.runs || {};
  var runIds = Object.keys(runs).sort(function (a, b) {
    return (runs[a].start_time || 0) - (runs[b].start_time || 0);
  });
  var COLORS = ["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed","#0891b2","#db2777","#65a30d","#4f46e5","#ea580c","#0d9488","#b45309"];
  var runColors = {};
  runIds.forEach(function (id, i) { runColors[id] = COLORS[i % COLORS.length]; });
  var visible = {};
  runIds.forEach(function (id) { visible[id] = true; });

  var metricSet = {};
  runIds.forEach(function (id) {
    var m = runs[id].metrics || {};
    Object.keys(m).forEach(function (k) { metricSet[k] = true; });
  });
  var metricList = Object.keys(metricSet).sort();

  document.getElementById('subtitle').textContent =
    runIds.length + ' run' + (runIds.length === 1 ? '' : 's') + ' · ' +
    metricList.length + ' metric' + (metricList.length === 1 ? '' : 's');
  if (DATA.generated_at) {
    var d = new Date(DATA.generated_at * 1000);
    document.getElementById('generated').textContent = 'updated ' + d.toLocaleString();
  }

  var runsListEl = document.getElementById('runs-list');
  runIds.forEach(function (id) {
    var r = runs[id];
    var item = document.createElement('div');
    item.className = 'run-item';
    var head = document.createElement('label');
    head.className = 'run-head';
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.dataset.runId = id;
    cb.addEventListener('change', function () {
      visible[id] = cb.checked;
      renderCharts();
    });
    var sw = document.createElement('span');
    sw.className = 'run-swatch';
    sw.style.background = runColors[id];
    var nm = document.createElement('span');
    nm.className = 'run-name';
    nm.textContent = r.name || id;
    nm.title = r.name || id;
    head.appendChild(cb); head.appendChild(sw); head.appendChild(nm);
    item.appendChild(head);
    if (r.start_time) {
      var meta = document.createElement('div');
      meta.className = 'run-meta';
      var ds = new Date(r.start_time * 1000);
      var dur = r.end_time ? fmtDuration(r.end_time - r.start_time) : 'running';
      meta.textContent = ds.toLocaleString() + ' · ' + dur;
      item.appendChild(meta);
    }
    var cfg = r.config || {};
    var cfgKeys = Object.keys(cfg);
    if (cfgKeys.length) {
      var cdiv = document.createElement('div');
      cdiv.className = 'run-config';
      cfgKeys.slice(0, 8).forEach(function (k) {
        var s = document.createElement('span');
        s.textContent = k + ': ' + formatCfg(cfg[k]);
        s.title = k + ': ' + String(cfg[k]);
        cdiv.appendChild(s);
      });
      if (cfgKeys.length > 8) {
        var more = document.createElement('span');
        more.textContent = '… +' + (cfgKeys.length - 8) + ' more';
        cdiv.appendChild(more);
      }
      item.appendChild(cdiv);
    }
    runsListEl.appendChild(item);
  });

  document.getElementById('bulk-toggle').addEventListener('click', function () {
    var boxes = runsListEl.querySelectorAll('input[type=checkbox]');
    var anyOn = false;
    boxes.forEach(function (b) { if (b.checked) anyOn = true; });
    boxes.forEach(function (b) {
      b.checked = !anyOn;
      visible[b.dataset.runId] = b.checked;
    });
    renderCharts();
  });

  function formatCfg(v) {
    if (v === null || v === undefined) return 'null';
    if (typeof v === 'number') return fmtNum(v);
    if (typeof v === 'string') return v.length > 28 ? v.slice(0, 28) + '…' : v;
    if (typeof v === 'boolean') return String(v);
    try { var s = JSON.stringify(v); return s.length > 28 ? s.slice(0, 28) + '…' : s; }
    catch (e) { return String(v); }
  }

  function fmtDuration(sec) {
    if (sec < 60) return sec.toFixed(1) + 's';
    if (sec < 3600) return (sec / 60).toFixed(1) + 'm';
    return (sec / 3600).toFixed(2) + 'h';
  }

  function fmtNum(v) {
    if (!isFinite(v)) return String(v);
    if (v === 0) return '0';
    var a = Math.abs(v);
    if (a >= 10000 || a < 0.001) return v.toExponential(2);
    if (Number.isInteger(v)) return v.toString();
    return parseFloat(v.toPrecision(4)).toString();
  }

  function niceNum(range, round) {
    var exp = Math.floor(Math.log10(range));
    var frac = range / Math.pow(10, exp);
    var nf;
    if (round) {
      if (frac < 1.5) nf = 1;
      else if (frac < 3) nf = 2;
      else if (frac < 7) nf = 5;
      else nf = 10;
    } else {
      if (frac <= 1) nf = 1;
      else if (frac <= 2) nf = 2;
      else if (frac <= 5) nf = 5;
      else nf = 10;
    }
    return nf * Math.pow(10, exp);
  }

  function niceTicks(min, max, n) {
    if (!isFinite(min) || !isFinite(max)) return { ticks: [0, 1], min: 0, max: 1 };
    if (min === max) { min = min - 0.5; max = max + 0.5; }
    var range = niceNum(max - min, false);
    var step = niceNum(range / Math.max(1, n - 1), true);
    var niceMin = Math.floor(min / step) * step;
    var niceMax = Math.ceil(max / step) * step;
    var ticks = [];
    for (var v = niceMin; v <= niceMax + step / 2; v += step) ticks.push(Math.round(v / step) * step);
    return { ticks: ticks, min: niceMin, max: niceMax };
  }

  var chartsEl = document.getElementById('charts');
  var tooltip = document.getElementById('tooltip');
  var SVGNS = 'http://www.w3.org/2000/svg';
  var logScale = {};
  var collapsed = {};

  function groupForKey(key) {
    var idx = key.indexOf('/');
    return idx > 0 ? key.slice(0, idx) : '';
  }

  function buildGroups() {
    var groups = {};
    metricList.forEach(function (key) {
      var g = groupForKey(key);
      if (!groups[g]) groups[g] = [];
      groups[g].push(key);
    });
    return groups;
  }

  function renderCharts() {
    chartsEl.innerHTML = '';
    if (metricList.length === 0) {
      var e = document.createElement('div');
      e.className = 'empty';
      e.textContent = 'No metrics logged yet. Call wandb.log({...}) to start.';
      chartsEl.appendChild(e);
      return;
    }
    var groups = buildGroups();
    var groupNames = Object.keys(groups).sort(function (a, b) {
      if (a === '') return -1;
      if (b === '') return 1;
      return a.localeCompare(b);
    });
    groupNames.forEach(function (gname) {
      var section = document.createElement('section');
      section.className = 'metric-group' + (collapsed[gname] ? ' collapsed' : '');
      var header = document.createElement('div');
      header.className = 'metric-group-header';
      var caret = document.createElement('span');
      caret.className = 'caret';
      caret.textContent = '▾';
      var h2 = document.createElement('h2');
      h2.textContent = gname || 'metrics';
      var count = document.createElement('span');
      count.className = 'count';
      count.textContent = groups[gname].length;
      header.appendChild(caret); header.appendChild(h2); header.appendChild(count);
      header.addEventListener('click', function () {
        collapsed[gname] = !collapsed[gname];
        section.classList.toggle('collapsed', collapsed[gname]);
      });
      section.appendChild(header);
      var grid = document.createElement('div');
      grid.className = 'charts-grid';
      groups[gname].sort().forEach(function (key) {
        grid.appendChild(buildChart(key));
      });
      section.appendChild(grid);
      chartsEl.appendChild(section);
    });
  }

  function buildChart(key) {
    var container = document.createElement('div');
    container.className = 'chart-container';
    var h = document.createElement('h3');
    var keySpan = document.createElement('span');
    keySpan.className = 'key';
    keySpan.textContent = key;
    keySpan.title = key;
    var logBtn = document.createElement('button');
    logBtn.className = 'log-toggle' + (logScale[key] ? ' active' : '');
    logBtn.textContent = 'log';
    logBtn.title = 'toggle log Y-axis';
    logBtn.addEventListener('click', function (ev) {
      ev.stopPropagation();
      logScale[key] = !logScale[key];
      var newChart = buildChart(key);
      container.parentNode.replaceChild(newChart, container);
    });
    h.appendChild(keySpan); h.appendChild(logBtn);
    container.appendChild(h);

    var isLog = !!logScale[key];

    var series = [];
    runIds.forEach(function (id) {
      if (!visible[id]) return;
      var m = runs[id].metrics || {};
      if (!m[key] || m[key].length === 0) return;
      var sorted = m[key].slice().sort(function (a, b) { return a[0] - b[0]; });
      if (isLog) sorted = sorted.filter(function (p) { return p[1] > 0; });
      if (sorted.length === 0) return;
      series.push({ id: id, name: runs[id].name || id, color: runColors[id], data: sorted });
    });

    var W = 520, H = 280;
    var pad = { l: 58, r: 14, t: 10, b: 30 };
    var innerW = W - pad.l - pad.r;
    var innerH = H - pad.t - pad.b;

    var svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('class', 'chart');
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    if (series.length === 0) {
      var txt = document.createElementNS(SVGNS, 'text');
      txt.setAttribute('x', W / 2);
      txt.setAttribute('y', H / 2);
      txt.setAttribute('text-anchor', 'middle');
      txt.setAttribute('fill', '#9ca3af');
      txt.setAttribute('font-size', '12');
      txt.textContent = 'no visible runs';
      svg.appendChild(txt);
      container.appendChild(svg);
      return container;
    }

    var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    series.forEach(function (s) {
      s.data.forEach(function (p) {
        if (p[0] < xMin) xMin = p[0];
        if (p[0] > xMax) xMax = p[0];
        if (p[1] < yMin) yMin = p[1];
        if (p[1] > yMax) yMax = p[1];
      });
    });
    if (xMin === xMax) { xMax = xMin + 1; }
    if (yMin === yMax) { var pad0 = Math.max(1, Math.abs(yMin) * 0.1); yMin -= pad0; yMax += pad0; }

    var yTicks, yLo, yHi;
    if (isLog) {
      var lo = Math.floor(Math.log10(yMin));
      var hi = Math.ceil(Math.log10(yMax));
      if (lo === hi) hi = lo + 1;
      yLo = Math.pow(10, lo); yHi = Math.pow(10, hi);
      yTicks = [];
      for (var e = lo; e <= hi; e++) yTicks.push(Math.pow(10, e));
    } else {
      var yN = niceTicks(yMin, yMax, 6);
      yLo = yN.min; yHi = yN.max; yTicks = yN.ticks;
    }
    var xN = niceTicks(xMin, xMax, 6);

    function sx(x) { return pad.l + (x - xMin) / (xMax - xMin) * innerW; }
    function sy(y) {
      if (isLog) {
        var lyMin = Math.log10(yLo), lyMax = Math.log10(yHi);
        return pad.t + (1 - (Math.log10(y) - lyMin) / (lyMax - lyMin)) * innerH;
      }
      return pad.t + (1 - (y - yLo) / (yHi - yLo)) * innerH;
    }

    yTicks.forEach(function (v) {
      var y = sy(v);
      if (y < pad.t - 0.5 || y > pad.t + innerH + 0.5) return;
      var g = document.createElementNS(SVGNS, 'line');
      g.setAttribute('x1', pad.l); g.setAttribute('x2', pad.l + innerW);
      g.setAttribute('y1', y); g.setAttribute('y2', y);
      g.setAttribute('class', 'grid');
      svg.appendChild(g);
      var t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('x', pad.l - 6); t.setAttribute('y', y + 3);
      t.setAttribute('text-anchor', 'end');
      t.setAttribute('class', 'tick');
      t.textContent = fmtNum(v);
      svg.appendChild(t);
    });

    xN.ticks.forEach(function (v) {
      if (v < xMin - 1e-9 || v > xMax + 1e-9) return;
      var x = sx(v);
      var t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('x', x); t.setAttribute('y', pad.t + innerH + 16);
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('class', 'tick');
      t.textContent = fmtNum(v);
      svg.appendChild(t);
    });

    var ax1 = document.createElementNS(SVGNS, 'line');
    ax1.setAttribute('x1', pad.l); ax1.setAttribute('x2', pad.l + innerW);
    ax1.setAttribute('y1', pad.t + innerH); ax1.setAttribute('y2', pad.t + innerH);
    ax1.setAttribute('class', 'axis');
    svg.appendChild(ax1);
    var ax2 = document.createElementNS(SVGNS, 'line');
    ax2.setAttribute('x1', pad.l); ax2.setAttribute('x2', pad.l);
    ax2.setAttribute('y1', pad.t); ax2.setAttribute('y2', pad.t + innerH);
    ax2.setAttribute('class', 'axis');
    svg.appendChild(ax2);

    series.forEach(function (s) {
      var d = '';
      s.data.forEach(function (p, i) {
        d += (i === 0 ? 'M' : 'L') + sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1);
      });
      var path = document.createElementNS(SVGNS, 'path');
      path.setAttribute('d', d);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', s.color);
      path.setAttribute('stroke-width', '1.75');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('stroke-linecap', 'round');
      svg.appendChild(path);
    });

    var cursor = document.createElementNS(SVGNS, 'line');
    cursor.setAttribute('class', 'cursor');
    cursor.setAttribute('y1', pad.t);
    cursor.setAttribute('y2', pad.t + innerH);
    svg.appendChild(cursor);
    var markers = [];
    series.forEach(function (s) {
      var c = document.createElementNS(SVGNS, 'circle');
      c.setAttribute('r', '3.5');
      c.setAttribute('fill', s.color);
      c.setAttribute('stroke', '#fff');
      c.setAttribute('stroke-width', '1.5');
      c.style.display = 'none';
      svg.appendChild(c);
      markers.push(c);
    });

    function hideCursor() {
      cursor.style.display = 'none';
      markers.forEach(function (m) { m.style.display = 'none'; });
      tooltip.style.display = 'none';
    }

    svg.addEventListener('mousemove', function (ev) {
      var rect = svg.getBoundingClientRect();
      var scale = rect.width / W;
      var mx = (ev.clientX - rect.left) / scale;
      if (mx < pad.l || mx > pad.l + innerW) { hideCursor(); return; }
      var xVal = xMin + (mx - pad.l) / innerW * (xMax - xMin);
      cursor.style.display = '';
      cursor.setAttribute('x1', mx); cursor.setAttribute('x2', mx);
      var rows = [];
      series.forEach(function (s, i) {
        var nearest = s.data[0], bestD = Math.abs(s.data[0][0] - xVal);
        for (var k = 1; k < s.data.length; k++) {
          var d = Math.abs(s.data[k][0] - xVal);
          if (d < bestD) { bestD = d; nearest = s.data[k]; }
        }
        markers[i].style.display = '';
        markers[i].setAttribute('cx', sx(nearest[0]));
        markers[i].setAttribute('cy', sy(nearest[1]));
        rows.push({ name: s.name, color: s.color, val: nearest[1], step: nearest[0] });
      });
      rows.sort(function (a, b) { return b.val - a.val; });
      var step = rows[0].step;
      var html = '<div class="ttstep">step ' + fmtNum(step) + '</div>';
      rows.forEach(function (r) {
        html += '<div class="ttrow"><span class="sw" style="background:' + r.color + '"></span>'
             + '<span class="nm">' + escapeHtml(r.name) + '</span>'
             + '<span style="margin-left:auto">' + fmtNum(r.val) + '</span></div>';
      });
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      var vw = window.innerWidth, vh = window.innerHeight;
      var tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
      var tx = ev.clientX + 14, ty = ev.clientY + 14;
      if (tx + tw > vw - 8) tx = ev.clientX - tw - 14;
      if (ty + th > vh - 8) ty = ev.clientY - th - 14;
      tooltip.style.left = tx + 'px';
      tooltip.style.top = ty + 'px';
    });
    svg.addEventListener('mouseleave', hideCursor);

    container.appendChild(svg);
    return container;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c];
    });
  }

  renderCharts();
})();
</script>
</body>
</html>
"""
