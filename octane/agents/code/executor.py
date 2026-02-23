"""Executor — writes code to a temp file, installs deps, runs it in subprocess.

Isolation model:
  - Each execution gets a fresh tempdir (cleaned up after)
  - pip installs go into that tempdir only (--target)
  - Hard timeout: 30 seconds max
  - stdout + stderr captured separately
  - Returns dict with stdout, stderr, exit_code, duration_ms
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import structlog

logger = structlog.get_logger().bind(component="code.executor")

TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 4000  # Truncate runaway output


class Executor:
    """Runs Python code in an isolated subprocess with optional pip deps."""

    async def run(self, code: str, requirements: list[str] | None = None) -> dict:
        """
        Execute code string. Returns:
            {
                "stdout": str,
                "stderr": str,
                "exit_code": int,
                "duration_ms": float,
                "truncated": bool,
            }
        """
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="octane_exec_") as tmpdir:
            # Write code to file
            code_path = os.path.join(tmpdir, "solution.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Install requirements if any
            if requirements:
                install_result = await self._install_deps(requirements, tmpdir)
                if install_result["exit_code"] != 0:
                    return {
                        "stdout": "",
                        "stderr": f"Dependency install failed:\n{install_result['stderr']}",
                        "exit_code": 1,
                        "duration_ms": (time.monotonic() - start) * 1000,
                        "truncated": False,
                    }

            # Run the code
            env = os.environ.copy()
            if requirements:
                # Add our tmp site-packages to PYTHONPATH
                env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")

            result = await self._run_subprocess(
                [sys.executable, code_path],
                cwd=tmpdir,
                env=env,
            )

        duration_ms = (time.monotonic() - start) * 1000
        result["duration_ms"] = duration_ms
        logger.info(
            "execution_complete",
            exit_code=result["exit_code"],
            duration_ms=f"{duration_ms:.0f}",
            stdout_len=len(result["stdout"]),
        )
        return result

    async def _install_deps(self, requirements: list[str], target_dir: str) -> dict:
        """Install packages into target_dir using pip."""
        logger.info("installing_deps", packages=requirements)
        return await self._run_subprocess(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--target", target_dir, *requirements],
        )

    async def _run_subprocess(
        self,
        cmd: list[str],
        cwd: str | None = None,
        env: dict | None = None,
    ) -> dict:
        """Run a command, capture output, enforce timeout."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return {
                    "stdout": "",
                    "stderr": f"Execution timed out after {TIMEOUT_SECONDS}s",
                    "exit_code": 124,
                    "truncated": False,
                }

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            truncated = False

            if len(stdout) > MAX_OUTPUT_CHARS:
                stdout = stdout[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
                truncated = True
            if len(stderr) > MAX_OUTPUT_CHARS:
                stderr = stderr[:MAX_OUTPUT_CHARS] + "\n... [truncated]"

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": proc.returncode,
                "truncated": truncated,
            }
        except Exception as e:
            logger.error("subprocess_error", error=str(e))
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
                "truncated": False,
            }
