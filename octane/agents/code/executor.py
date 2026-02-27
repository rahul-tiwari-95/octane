"""Executor — writes code to a temp file, installs deps, runs it in subprocess.

Isolation model:
  - Each execution gets a fresh tempdir (cleaned up after)
  - pip installs go into that tempdir only (--target)
  - Hard timeout: 30 seconds max
  - stdout + stderr captured separately
  - Output files (charts, CSVs) are copied to ~/octane_output/<correlation_id>/
  - Returns dict with stdout, stderr, exit_code, duration_ms, output_files
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import structlog

logger = structlog.get_logger().bind(component="code.executor")

TIMEOUT_SECONDS = 120
MAX_OUTPUT_CHARS = 4000

# Where output files (charts, CSVs) are saved permanently
OUTPUT_ROOT = Path.home() / "octane_output"

# File extensions considered "output artifacts" worth saving
_ARTIFACT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".csv", ".json", ".txt", ".html"}


class Executor:
    """Runs Python code in an isolated subprocess with optional pip deps."""

    async def run(
        self,
        code: str,
        requirements: list[str] | None = None,
        correlation_id: str | None = None,
    ) -> dict:
        """
        Execute code string. Returns:
            {
                "stdout": str,
                "stderr": str,
                "exit_code": int,
                "duration_ms": float,
                "truncated": bool,
                "output_dir": str | None,   # path where artifacts were saved
                "output_files": list[str],  # filenames of saved artifacts
            }
        """
        start = time.monotonic()

        # Prepare permanent output dir for this execution
        run_id = correlation_id or f"run_{int(time.time() * 1000)}"
        output_dir = OUTPUT_ROOT / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Inject OUTPUT_DIR into the code so scripts can use it
        preamble = (
            f"import os as _os\n"
            f"OUTPUT_DIR = r'{output_dir}'\n"
            f"_os.makedirs(OUTPUT_DIR, exist_ok=True)\n\n"
        )
        code_with_preamble = preamble + code

        with tempfile.TemporaryDirectory(prefix="octane_exec_") as tmpdir:
            code_path = os.path.join(tmpdir, "solution.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code_with_preamble)

            if requirements:
                install_result = await self._install_deps(requirements, tmpdir)
                if install_result["exit_code"] != 0:
                    return {
                        "stdout": "",
                        "stderr": f"Dependency install failed:\n{install_result['stderr']}",
                        "exit_code": 1,
                        "duration_ms": (time.monotonic() - start) * 1000,
                        "truncated": False,
                        "output_dir": str(output_dir),
                        "output_files": [],
                    }

            env = os.environ.copy()
            if requirements:
                env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")

            result = await self._run_subprocess(
                [sys.executable, code_path],
                cwd=tmpdir,
                env=env,
            )

            # Collect any artifact files written to tmpdir or output_dir
            artifacts = self._collect_artifacts(tmpdir, output_dir)

        duration_ms = (time.monotonic() - start) * 1000
        result["duration_ms"] = duration_ms
        result["output_dir"] = str(output_dir) if artifacts else None
        result["output_files"] = artifacts

        logger.info(
            "execution_complete",
            exit_code=result["exit_code"],
            duration_ms=f"{duration_ms:.0f}",
            stdout_len=len(result["stdout"]),
            artifacts=len(artifacts),
            output_dir=str(output_dir) if artifacts else None,
        )
        return result

    async def _install_deps(self, requirements: list[str], target_dir: str) -> dict:
        """Install packages into target_dir using pip."""
        logger.info("installing_deps", packages=requirements)
        return await self._run_subprocess(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--target", target_dir, *requirements],
        )

    def _collect_artifacts(self, tmpdir: str, output_dir: Path) -> list[str]:
        """Collect artifact files from both tmpdir root and output_dir.

        Scripts may write to OUTPUT_DIR directly (already in output_dir) or
        to their cwd (tmpdir). Only the ROOT of tmpdir is scanned — subdirectories
        are ignored to avoid picking up pip-installed package data (e.g. matplotlib
        test CSVs, numpy validation files). OUTPUT_DIR is scanned recursively but
        only files at the top level are collected (scripts write to its root).
        Returns list of filenames found.
        """
        saved: list[str] = []

        # Files the script wrote directly to OUTPUT_DIR (already in place)
        for item in output_dir.iterdir():
            if item.is_file() and item.suffix.lower() in _ARTIFACT_EXTENSIONS:
                saved.append(item.name)
                logger.info("artifact_found", file=item.name, path=str(item))

        # Files the script wrote to cwd root (tmpdir) — copy them over.
        # Only scan the root of tmpdir, NOT subdirectories, to avoid pip-installed
        # package data (e.g. matplotlib/mpl-data, numpy test sets).
        tmpdir_path = Path(tmpdir)
        for item in tmpdir_path.iterdir():
            if item.is_file() and item.suffix.lower() in _ARTIFACT_EXTENSIONS:
                if item.name not in saved:
                    dst = output_dir / item.name
                    try:
                        shutil.copy2(item, dst)
                        saved.append(item.name)
                        logger.info("artifact_saved", file=item.name, path=str(dst))
                    except Exception as e:
                        logger.warning("artifact_copy_failed", file=item.name, error=str(e))
        return saved

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
