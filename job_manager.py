from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal


@dataclass
class SimulationJob:
    job_id: str
    config_path: str
    workspace_folder: str
    status: str = "pending"
    progress: int = 0
    total_steps: int = 0
    log_path: str = ""
    return_code: int | None = None


class SolverThread(QThread):
    output = Signal(str, str)
    progressChanged = Signal(str, int)
    statusChanged = Signal(str, str)
    finishedJob = Signal(object)

    def __init__(self, job: SimulationJob, command, cwd, env_updates=None, parent=None):
        super().__init__(parent)
        self.job = job
        self.command = list(command or [])
        self.cwd = str(cwd or "")
        self.env_updates = dict(env_updates or {})
        self._process = None
        self._cancel_requested = False
        self._progress_pattern = re.compile(r"Time steps completed\s*[:=]?\s*(\d+)", re.IGNORECASE)

    def cancel(self):
        self._cancel_requested = True
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except Exception:
            pass

    def stop(self):
        self.cancel()
        try:
            self.requestInterruption()
        except Exception:
            pass
        try:
            self.quit()
        except Exception:
            pass
        try:
            if QThread.currentThread() != self:
                self.wait()
        except Exception:
            pass

    def _emit_status(self, status):
        self.job.status = str(status)
        self.statusChanged.emit(self.job.job_id, self.job.status)

    def _emit_progress_from_line(self, line):
        if self.job.total_steps <= 0:
            return
        match = self._progress_pattern.search(str(line or ""))
        if not match:
            return
        try:
            step = int(match.group(1))
        except Exception:
            return
        pct = int(min(100, max(0, (step / max(1, self.job.total_steps)) * 100)))
        if pct == self.job.progress:
            return
        self.job.progress = pct
        self.progressChanged.emit(self.job.job_id, pct)

    def run(self):
        log_file = None
        try:
            log_path = Path(self.job.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.update(self.env_updates)

            self._emit_status("running")
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )

            for raw_line in iter(self._process.stdout.readline, ""):
                line = raw_line.rstrip("\n")
                if log_file is not None:
                    log_file.write(raw_line)
                    log_file.flush()
                self.output.emit(self.job.job_id, line)
                self._emit_progress_from_line(line)
                if self._cancel_requested:
                    break

            if self._process.stdout:
                self._process.stdout.close()
            return_code = self._process.wait()
            self.job.return_code = int(return_code)
            if self._cancel_requested:
                self._emit_status("stopped")
            elif return_code == 0:
                self.job.progress = 100
                self.progressChanged.emit(self.job.job_id, 100)
                self._emit_status("completed")
            else:
                self._emit_status(f"failed ({return_code})")
        except Exception as exc:
            self.job.return_code = -1
            self.output.emit(self.job.job_id, str(exc))
            self._emit_status("failed")
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            self.finishedJob.emit(self.job)


class JobManager(QObject):
    jobStarted = Signal(object)
    jobOutput = Signal(str, str)
    jobProgress = Signal(str, int)
    jobStatus = Signal(str, str)
    jobFinished = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._current_job = None

    def track_current_job(self):
        return self._current_job

    def run_job(self, *, config_path, workspace_folder, command, cwd, env_updates=None, total_steps=0):
        if self._thread is not None and self._thread.isRunning():
            return None
        job = self._build_job(config_path=config_path, workspace_folder=workspace_folder, total_steps=total_steps)
        thread = SolverThread(job, command=command, cwd=cwd, env_updates=env_updates, parent=self)
        thread.output.connect(self.jobOutput.emit)
        thread.progressChanged.connect(self.jobProgress.emit)
        thread.statusChanged.connect(self.jobStatus.emit)
        thread.finishedJob.connect(self._on_job_finished)
        self._current_job = job
        self._thread = thread
        self.jobStarted.emit(job)
        thread.start()
        return job

    def cancel_job(self):
        if self._thread is None:
            return
        self._thread.cancel()

    def stop(self):
        thread = self._thread
        self._current_job = None
        self._thread = None
        if thread is None:
            return
        try:
            thread.stop()
        except Exception:
            pass

    def _on_job_finished(self, job):
        self.jobFinished.emit(job)
        thread = self._thread
        self._current_job = None
        self._thread = None
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass

    def _build_job(self, *, config_path, workspace_folder, total_steps=0):
        workspace = Path(str(workspace_folder))
        jobs_dir = workspace / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        existing = []
        for path in jobs_dir.iterdir():
            name = path.name
            if path.is_dir() and name.startswith("job_"):
                suffix = name.split("_", 1)[-1]
                if suffix.isdigit():
                    existing.append(int(suffix))
        next_id = max(existing, default=0) + 1
        job_id = f"job_{next_id:03d}"
        job_dir = jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return SimulationJob(
            job_id=job_id,
            config_path=str(config_path),
            workspace_folder=str(workspace_folder),
            status="queued",
            progress=0,
            total_steps=max(0, int(total_steps or 0)),
            log_path=str(job_dir / "solver.log"),
        )
