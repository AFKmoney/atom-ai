"""Pause (SIGSTOP) MY probe PIDs while the live b4828 loop is in a multi-threaded phase.

Live torch uses 8 spinning OMP threads; any other busy process on a core cuts live
throughput ~38x (measured 77 -> 2 steps/s). The live stream-skip phase is
single-threaded, so my single-threaded probe runs only then. Only signals the PIDs
listed in the pid file (my own processes). Never touches the live loop.
"""
import glob, os, signal, sys, time, subprocess
PIDFILE = sys.argv[1]
LOG = "/workspace/exports/atom-ai/logs/ladrop_yield.log"
ROOT = "/workspace/exports/atom-ai/logs"

def my_pids():
    try:
        return [int(x) for x in open(PIDFILE).read().split() if x.strip()]
    except FileNotFoundError:
        return []

def alive(pg):
    # pg is a process-group id (setsid leader); alive if any member exists
    try:
        os.killpg(pg, 0)
        return True
    except Exception:
        return False

def live_busy():
    mine = set(my_pids())
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        pid_s, _, args = line.strip().partition(" ")
        pid = int(pid_s)
        try:
            if os.getpgid(pid) in mine:
                continue
        except Exception:
            continue
        if "ladrop" in args:
            continue
        if "chat_atom_native.py" in args and "lrd_b4828.pt" in args:
            return True
        if "run_atom_native.py" in args and "lrd_b4828.pt" in args:
            logs = [l for l in glob.glob(f"{ROOT}/train_d32_*_ms1M_lrd_b4828.log")]
            if not logs:
                return True
            cur = max(logs, key=os.path.getmtime)
            txt = open(cur, errors="replace").read()
            if "stream skip done" in txt:
                return True  # training / val / generation phase
            return False  # still skipping (single-threaded)
    return False

state = None
while True:
    pids = [p for p in my_pids() if alive(p)]
    if not pids:
        break
    busy = live_busy()
    want = "STOP" if busy else "CONT"
    for p in pids:
        try:
            os.killpg(p, signal.SIGSTOP if busy else signal.SIGCONT)
        except ProcessLookupError:
            pass
    if want != state:
        with open(LOG, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {want} pids={pids}\n")
        state = want
    time.sleep(0.5)
for p in my_pids():
    try: os.killpg(p, signal.SIGCONT)
    except Exception: pass
