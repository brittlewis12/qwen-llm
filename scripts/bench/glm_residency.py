"""Read-only macOS process/host observations for glm-workspace.py."""

import ctypes
import json
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


class Task(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("virtual_bytes", "resident_bytes", "user", "system", "threads_user", "threads_system")] + [
        (name, ctypes.c_int32) for name in
        ("policy", "faults", "pageins", "cow_faults", "messages_sent", "messages_received",
         "mach_calls", "unix_calls", "switches", "thread_count", "running_threads", "priority")]


LIB = ctypes.CDLL("libproc.dylib", use_errno=True)
LIB.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
LIB.proc_pid_rusage.restype = ctypes.c_int
LIB.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                            ctypes.c_void_p, ctypes.c_int]
LIB.proc_pidinfo.restype = ctypes.c_int


def usage(pid):
    result = Usage()
    if LIB.proc_pid_rusage(pid, 2, ctypes.byref(result)) != 0:
        return None
    values = {key: getattr(result, key) for key in
              ("pageins", "wired", "rss", "footprint", "disk_read", "disk_written", "user_time", "system_time")}
    task = Task()
    if LIB.proc_pidinfo(pid, 4, 0, ctypes.byref(task), ctypes.sizeof(task)) == ctypes.sizeof(task):
        values.update({key: getattr(task, key) for key in
                       ("faults", "cow_faults", "virtual_bytes", "switches")})
    return values


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
    def __init__(self):
        self.samples = []

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

    def finish(self, text):
        events = [json.loads(line.split("glm residency: ", 1)[1])
                  for line in text.splitlines() if "glm residency: " in line]
        if not events or not any(e["phase"] == "before_prefetch" for e in events) or events[-1]["phase"] != "after_decode":
            raise RuntimeError("missing complete native phase observations; use the instrumented engine commit")
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
            for process in ("qwen", "driver"):
                readings = [sample[process] for sample in observed if sample[process] is not None]
                phases[-1][process + "_ranges"] = {
                    key: {"min": min(values), "max": max(values), "delta": values[-1]-values[0]}
                    for key in set.intersection(*(set(row) for row in readings))
                    if (values := [row[key] for row in readings])} if readings else {}
        return {"events": events, "phases": phases,
                "sample_count": len(self.samples), "sample_guard_seconds": 1,
                "initial_host": self.samples[0] if self.samples else None,
                "final_host": self.samples[-1] if self.samples else None}
