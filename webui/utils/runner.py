# -*- coding: utf-8 -*-
"""
命令执行器
==========

负责执行训练命令，管理进程，捕获输出日志。
"""

import subprocess
import threading
import signal
import os
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class RunStatus:
    """运行状态"""

    running: bool = False
    exit_code: Optional[int] = None
    error: Optional[str] = None


class CommandRunner:
    """命令执行器，支持实时日志输出和进程管理"""

    def __init__(self, working_dir: str = None):
        """初始化执行器

        Args:
            working_dir: 工作目录，默认为项目根目录
        """
        if working_dir is None:
            working_dir = Path(__file__).parent.parent.parent
        self.working_dir = Path(working_dir)
        self.process: Optional[subprocess.Popen] = None
        self.status = RunStatus()
        self._log_callback: Optional[Callable[[str], None]] = None
        self._thread: Optional[threading.Thread] = None

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
        """构建训练/评估命令

        Args:
            mode: 运行模式 (unlearn/inject/edit/eval)
            model: 模型名称（HuggingFace 模型或配置名）
            trainer: 训练方法
            experiment: 实验模板
            task_name: 任务名称
            overrides: 参数覆盖字典
            eval_suite: 评测套件（eval 模式）
            model_path: 已保存模型路径（eval 模式）

        Returns:
            完整的命令行字符串
        """
        # eval 模式使用 eval.py
        if mode == "eval":
            cmd_parts = [
                "python",
                "src/eval.py",
                "--config-name=eval.yaml",
            ]

            # 添加评测套件
            if eval_suite:
                cmd_parts.append(f"eval={eval_suite}")

            # 添加模型路径或模型配置
            if model_path:
                cmd_parts.append(
                    f"model.model_args.pretrained_model_name_or_path={model_path}"
                )
            elif model:
                cmd_parts.append(f"model={model}")

            # 添加任务名称
            cmd_parts.append(f"task_name={task_name}")
        else:
            # 训练模式使用 train.py
            cmd_parts = [
                "python",
                "src/train.py",
                f"--config-name={mode}.yaml",
            ]

            # 添加实验模板
            if experiment and not experiment.startswith("("):
                cmd_parts.append(f"experiment={experiment}")

            # 添加模型
            if model:
                cmd_parts.append(f"model={model}")

            # 添加训练方法
            if trainer:
                cmd_parts.append(f"trainer={trainer}")

            # 添加任务名称
            cmd_parts.append(f"task_name={task_name}")

        # 添加参数覆盖
        if overrides:
            for key, value in overrides.items():
                if value is not None and value != "":
                    # 处理字符串值
                    if isinstance(value, str) and " " in value:
                        cmd_parts.append(f'{key}="{value}"')
                    else:
                        cmd_parts.append(f"{key}={value}")

        return " \\\n  ".join(cmd_parts)

    def run(
        self,
        command: str,
        log_callback: Callable[[str], None] = None,
        env: dict = None,
    ) -> bool:
        """执行命令

        Args:
            command: 要执行的命令
            log_callback: 日志回调函数，每行输出调用一次
            env: 额外的环境变量

        Returns:
            是否成功启动
        """
        if self.status.running:
            return False

        self._log_callback = log_callback
        self.status = RunStatus(running=True)

        # 准备环境变量
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # 启动后台线程执行命令
        self._thread = threading.Thread(
            target=self._run_process, args=(command, run_env), daemon=True
        )
        self._thread.start()

        return True

    def _run_process(self, command: str, env: dict):
        """后台执行进程"""
        try:
            # 将多行命令合并为单行
            cmd_single = command.replace("\\\n", " ").replace("  ", " ")

            self._log(f"[INFO] 工作目录: {self.working_dir}")
            self._log(f"[INFO] 执行命令: {cmd_single}\n")

            self.process = subprocess.Popen(
                cmd_single,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.working_dir),
                env=env,
                text=True,
                bufsize=1,
            )

            # 实时读取输出
            for line in iter(self.process.stdout.readline, ""):
                if not self.status.running:
                    break
                self._log(line.rstrip("\n"))

            self.process.wait()
            self.status.exit_code = self.process.returncode

            if self.status.exit_code == 0:
                self._log("\n[SUCCESS] 命令执行完成")
            else:
                self._log(f"\n[ERROR] 命令执行失败，退出码: {self.status.exit_code}")

        except Exception as e:
            self.status.error = str(e)
            self._log(f"\n[ERROR] 执行异常: {e}")
        finally:
            self.status.running = False
            self.process = None

    def _log(self, message: str):
        """发送日志"""
        if self._log_callback:
            self._log_callback(message)

    def stop(self) -> bool:
        """停止当前运行的命令

        Returns:
            是否成功停止
        """
        if not self.status.running or self.process is None:
            return False

        try:
            self._log("\n[INFO] 正在停止进程...")
            # 发送 SIGINT 信号
            self.process.send_signal(signal.SIGINT)
            # 等待进程结束
            self.process.wait(timeout=5)
            self.status.running = False
            self._log("[INFO] 进程已停止")
            return True
        except subprocess.TimeoutExpired:
            # 强制终止
            self.process.kill()
            self.status.running = False
            self._log("[INFO] 进程已强制终止")
            return True
        except Exception as e:
            self._log(f"[ERROR] 停止失败: {e}")
            return False

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.status.running
