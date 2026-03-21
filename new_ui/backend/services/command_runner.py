"""
命令执行服务 - 适配自 webui/utils/runner.py，增加 SSE 日志队列。
"""

import os
import signal
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional


@dataclass
class RunStatus:
    running: bool = False
    exit_code: Optional[int] = None
    error: Optional[str] = None


class CommandRunner:

    def __init__(self, working_dir: str = None):
        if working_dir is None:
            working_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.working_dir = Path(working_dir)
        self.process: Optional[subprocess.Popen] = None
        self.status = RunStatus()
        self._thread: Optional[threading.Thread] = None
        self.log_lines: Deque[str] = deque(maxlen=50000)
        self._log_event = threading.Event()

    def build_command(
        self,
        mode: str,
        model: str = None,
        trainer: str = None,
        experiment: str = None,
        task_name: str = "experiment",
        overrides: dict = None,
        eval_suite: str = None,
        model_path: str = None,
    ) -> str:
        if mode == "eval":
            parts = ["python", "src/eval.py", "--config-name=eval.yaml"]
            if eval_suite:
                parts.append(f"eval={eval_suite}")
            if model_path:
                parts.append(f"model.model_args.pretrained_model_name_or_path={model_path}")
            elif model:
                parts.append(f"model={model}")
            parts.append(f"task_name={task_name}")
        else:
            parts = ["python", "src/train.py", f"--config-name={mode}.yaml"]
            if experiment and not experiment.startswith("("):
                parts.append(f"experiment={experiment}")
            if model:
                parts.append(f"model={model}")
            if trainer:
                parts.append(f"trainer={trainer}")
            parts.append(f"task_name={task_name}")

        if overrides:
            for k, v in overrides.items():
                if v is not None and v != "":
                    parts.append(f'{k}="{v}"' if isinstance(v, str) and " " in v else f"{k}={v}")

        return " \\\n  ".join(parts)

    def run(self, command: str, env: dict = None) -> bool:
        if self.status.running:
            return False
        self.log_lines.clear()
        self.status = RunStatus(running=True)
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        self._thread = threading.Thread(target=self._run_process, args=(command, run_env), daemon=True)
        self._thread.start()
        return True

    def _run_process(self, command: str, env: dict):
        try:
            cmd = command.replace("\\\n", " ").replace("  ", " ")
            self._push(f"[INFO] Working dir: {self.working_dir}")
            self._push(f"[INFO] Command: {cmd}\n")
            self.process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=str(self.working_dir), env=env, text=True, bufsize=1,
            )
            for line in iter(self.process.stdout.readline, ""):
                if not self.status.running:
                    break
                self._push(line.rstrip("\n"))
            self.process.wait()
            self.status.exit_code = self.process.returncode
            self._push(f"\n[{'SUCCESS' if self.status.exit_code == 0 else 'ERROR'}] exit code {self.status.exit_code}")
        except Exception as e:
            self.status.error = str(e)
            self._push(f"\n[ERROR] {e}")
        finally:
            self.status.running = False
            self.process = None

    def _push(self, msg: str):
        self.log_lines.append(msg)
        self._log_event.set()

    def stop(self) -> bool:
        if not self.status.running or self.process is None:
            return False
        try:
            self._push("\n[INFO] Stopping...")
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        finally:
            self.status.running = False
            self._push("[INFO] Stopped")
        return True

    def wait_for_log(self, timeout: float = 1.0) -> bool:
        fired = self._log_event.wait(timeout=timeout)
        self._log_event.clear()
        return fired
