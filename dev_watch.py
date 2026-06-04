import os
import signal
import subprocess
import sys
import time


WATCH_EXTS = {".py", ".yml", ".yaml", ".json", ".txt", ".csv"}
IGNORE_DIRS = {".git", "__pycache__", "results", "CPD-main_v0", "CPD-main_v1", "CPD-main_v2", ".venv"}
POLL_SECONDS = 0.5


def _iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in WATCH_EXTS:
                yield os.path.join(dirpath, name)


def _snapshot(root):
    snap = {}
    for path in _iter_files(root):
        try:
            snap[path] = os.path.getmtime(path)
        except OSError:
            continue
    return snap


def _diff(old, new):
    if old.keys() != new.keys():
        return True
    for path, mtime in new.items():
        if old.get(path) != mtime:
            return True
    return False


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "main_window.py"]
    print(f"[watch] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    snap = _snapshot(root)
    try:
        while True:
            time.sleep(POLL_SECONDS)
            new_snap = _snapshot(root)
            if _diff(snap, new_snap):
                print("[watch] Change detected. Restarting...")
                snap = new_snap
                if proc.poll() is None:
                    try:
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=2)
                    except Exception:
                        proc.kill()
                proc = subprocess.Popen(cmd)
    except KeyboardInterrupt:
        print("\n[watch] Stopping.")
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
