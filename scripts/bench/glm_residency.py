"""Read-only macOS process/host observations for glm-workspace.py."""

import ctypes
import os
import re
import subprocess
import time


class Usage(ctypes.Structure):
    # sys/resource.h rusage_info_v2, including the child fields before disk I/O.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle", "interrupts", "pageins", "wired",
        "rss", "footprint", "start", "exit", "child_user", "child_system",
        "child_idle", "child_interrupts", "child_pageins", "child_elapsed",
        "disk_read", "disk_written")]


LIB = ctypes.CDLL("libproc.dylib", use_errno=True)
LIB.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
LIB.proc_pid_rusage.restype = ctypes.c_int


def usage(pid):
    result = Usage()
    if LIB.proc_pid_rusage(pid, 2, ctypes.byref(result)) != 0:
        return None
    return {key: getattr(result, key) for key in
            ("pageins", "wired", "rss", "footprint", "disk_read", "disk_written")}


def host():
    text = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+) bytes", text)[1])
    values = {key: int(value) for key, value in
              re.findall(r'^"?([^":\n]+)"?:\s+(\d+)\.', text, re.M)}
    gauges = ("Pages free", "Pages active", "Pages inactive", "Pages speculative",
              "Pages wired down", "File-backed pages", "Anonymous pages",
              "Pages stored in compressor", "Pages occupied by compressor")
    counters = ("Pageins", "Pageouts", "Swapins", "Swapouts", "Compressions", "Decompressions")
    return {"page_bytes": page,
            **{key: values[key] * page for key in gauges},
            **{key: values[key] for key in counters}}


class Observer:
    def __init__(self, memory_map=False):
        self.samples = []
        self.memory_map = memory_map
        self.mapping = None

    def sample(self, rows, log):
        now = time.time()
        records = [line.split(maxsplit=3) for line in rows.splitlines()]
        records = [row for row in records if len(row) == 4]
        compilers = [int(pid) for pid, _, _, command in records
                     if command.rsplit("/", 1)[-1].strip("()") == "rustc"]
        children = [int(pid) for pid, ppid, _, command in records
                    if int(ppid) == os.getpid() and command.rsplit("/", 1)[-1] == "qwen"]
        pid = children[0] if len(children) == 1 else None
        compiler_usage = [entry for pid in compilers if (entry := usage(pid)) is not None]
        self.samples.append({"unix_seconds": now, "host": host(),
            "compiler_count": len(compilers),
            "compiler_rss_bytes": sum(int(rss) * 1024 for pid, _, rss, _ in records
                                      if int(pid) in compilers),
            "compiler_footprint_bytes": sum(u["footprint"] for u in compiler_usage),
            "driver": usage(os.getpid()), "qwen": usage(pid) if pid else None})
        if (self.memory_map and self.mapping is None and pid and log and log.exists()
                and '"phase":"after_session"' in log.read_text()):
            started = time.monotonic()
            result = subprocess.run(["vmmap", "-w", str(pid)], capture_output=True,
                                    text=True, timeout=30)
            lines = result.stdout.splitlines()
            # Keep compact mapping rows and the OS summary, without host paths.
            gguf = [line[:line.rfind(" /")] + " " + line.rsplit("/", 1)[-1]
                    for line in lines if ".gguf" in line and " /" in line]
            summary = lines[next((i for i, line in enumerate(lines)
                                  if "REGION TYPE" in line and "VIRTUAL" in line), len(lines)):]
            self.mapping = {"status": result.returncode,
                "wall_ms": (time.monotonic()-started)*1000,
                "gguf_rows": gguf, "summary": summary[-100:],
                "stderr": result.stderr[-500:]}

    def finish(self, text):
        events = [__import__("json").loads(line.split("glm residency: ", 1)[1])
                  for line in text.splitlines() if "glm residency: " in line]
        phases = []
        for before, after in zip(events, events[1:]):
            observed = [sample for sample in self.samples
                        if before["unix_seconds"]-1 <= sample["unix_seconds"]
                        <= after["unix_seconds"]+1]
            ranges = {}
            if observed:
                for key in observed[0]["host"]:
                    values = [sample["host"][key] for sample in observed]
                    ranges[key] = {"min": min(values), "max": max(values),
                                   "delta": values[-1]-values[0]}
            phases.append({"from": before["phase"], "to": after["phase"],
                "wall_ms": (after["unix_seconds"]-before["unix_seconds"])*1000,
                "process_pageins": after["process"]["pageins"]-before["process"]["pageins"],
                "process_disk_read_bytes": after["process"]["disk_read_bytes"]-before["process"]["disk_read_bytes"],
                "host_ranges": ranges, "samples": len(observed),
                "compiler_rss_peak_bytes": max((s["compiler_rss_bytes"] for s in observed), default=0),
                "compiler_footprint_peak_bytes": max((s["compiler_footprint_bytes"] for s in observed), default=0)})
        return {"events": events, "phases": phases, "mapping": self.mapping,
                "sample_count": len(self.samples), "sample_guard_seconds": 1,
                "initial_host": self.samples[0] if self.samples else None,
                "final_host": self.samples[-1] if self.samples else None}
