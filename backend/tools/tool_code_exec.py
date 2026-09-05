"""
backend/tools/tool_code_exec.py

Local Python execution tool for the Sovereign AI Workbench.

This is intentionally a small, offline-only execution helper for local code
experiments and agent tasks. It is NOT a complete OS-level sandbox for
hostile or untrusted code.

The executor in backend/agent/executer.py calls tools through a callable API
that expects a function signature compatible with tool(**args). This module
therefore exposes execute_python_code(code, **kwargs), which is the public
entry point used by the agent and can also be called directly.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

__all__ = ["execute_python_code", "tool"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CODE_EXECUTION_ROOT = PROJECT_ROOT / "sandbox" / "code_execution"
CODE_EXECUTION_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30.0
_MAX_TIMEOUT = 30.0

_DEFAULT_MAX_CODE_SIZE = 200_000
_DEFAULT_MAX_OUTPUT_SIZE = 100_000


# ---------------------------------------------------------------------------
# Safe environment variables
# ---------------------------------------------------------------------------

_ALLOWED_USER_ENV_KEYS = {
    "PATH",
    "SystemRoot",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
    "PYTHONIOENCODING",
    "PYTHONHASHSEED",
}


# ---------------------------------------------------------------------------
# Defense-in-depth safety patterns
# ---------------------------------------------------------------------------

DANGEROUS_PATTERNS: List[str] = [
    r"\bsubprocess\b",
    r"\bos\.system\s*\(",
    r"\bos\.popen\s*\(",
    r"\bos\.spawn\w*\s*\(",
    r"\bos\.exec\w*\s*\(",
    r"\bsubprocess\.(run|popen|check_output|check_call|call)\s*\(",
    r"\bmultiprocessing\b",
    r"\bthreading\b",
    r"\bctypes\b",
    r"\bpty\b",
    r"\bsocket\b",
    r"\brequests\b",
    r"\burllib\.request\b",
    r"\bhttpx\b",
    r"\bhttp\.client\b",
    r"\bwebbrowser\b",
    r"\bopen\s*\(\s*[\"'](?:[a-zA-Z]:)?[/\\]",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(value: str, max_size: int) -> tuple[str, bool]:
    """Limit output size while preserving a truncation flag."""
    if value is None:
        return "", False

    if len(value) <= max_size:
        return value, False

    return value[:max_size], True


def _looks_dangerous(code: str) -> List[str]:
    """
    Return a list of obvious dangerous patterns found in the submitted code.

    This is only defense-in-depth. It is NOT a complete security boundary.
    Matching is case-insensitive.
    """
    matches: List[str] = []

    for pattern in DANGEROUS_PATTERNS:
        try:
            if re.search(pattern, code, flags=re.IGNORECASE):
                matches.append(pattern)
        except re.error:
            # Defensive handling in case a future pattern is malformed.
            continue

    return matches


def _validate_positive_number(
    value: Any,
    name: str,
    default: float,
) -> Optional[float]:
    """
    Validate a positive finite numeric argument.

    Returns:
        Validated float value, or None if invalid.

    Rejects:
        - bool
        - non-numeric values
        - zero
        - negative values
        - NaN
        - infinity
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    if number <= 0:
        return None

    return number


def _validate_positive_integer(
    value: Any,
    name: str,
    default: int,
) -> Optional[int]:
    """
    Validate a positive integer limit.

    Fractional values are rejected instead of silently truncated.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    if number <= 0:
        return None

    if not number.is_integer():
        return None

    return int(number)


def _rejected_result(
    reason: str,
    timeout: float,
    blocked_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a stable rejected result dictionary."""
    return {
        "success": False,
        "status": "rejected",
        "returncode": None,
        "stdout": "",
        "stderr": reason,
        "execution_time": 0.0,
        "timeout": timeout,
        "truncated": {
            "stdout": False,
            "stderr": False,
        },
        "blocked_patterns": blocked_patterns or [],
    }


def _minimal_environment(
    user_env: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """
    Build a minimal environment for the executed subprocess.

    Only explicitly allowed variables are inherited from the parent process.
    User-provided environment variables are also restricted to the same
    whitelist.
    """
    env: Dict[str, str] = {}

    for key in _ALLOWED_USER_ENV_KEYS:
        value = os.environ.get(key)

        if value is not None:
            env[key] = value

    # Stable Python execution behavior.
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    if isinstance(user_env, dict):
        for key, value in user_env.items():
            if (
                key in _ALLOWED_USER_ENV_KEYS
                and isinstance(value, str)
            ):
                env[key] = value

    return env


def _resolve_within_sandbox(
    path_value: Any,
    name: str,
) -> Optional[Path]:
    """
    Resolve a path and ensure it remains inside sandbox/code_execution.

    Returns:
        Resolved Path if valid, otherwise None.
    """
    if path_value is None:
        return None

    try:
        candidate = (
            Path(str(path_value))
            .expanduser()
            .resolve()
        )
    except (TypeError, ValueError, OSError):
        return None

    sandbox_root = CODE_EXECUTION_ROOT.resolve()

    try:
        candidate.relative_to(sandbox_root)
    except ValueError:
        return None

    return candidate


# ---------------------------------------------------------------------------
# Public execution function
# ---------------------------------------------------------------------------

def execute_python_code(
    code: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute Python code in a child process under sandbox/code_execution/.

    This function is designed for local/offline agent execution.

    It is NOT a complete OS-level sandbox for hostile or untrusted code.

    Args:
        code:
            Python source code.

        timeout:
            Maximum execution time in seconds.
            Fractional values are supported.
            Maximum allowed value is 30 seconds.

        max_code_size:
            Maximum submitted code length.

        max_output_size:
            Maximum stdout/stderr length retained.

        cwd:
            Optional working directory. Must be inside sandbox/code_execution.

        env:
            Optional environment mapping. Only whitelisted keys are accepted.

        workspace:
            Optional workspace directory. Must be inside sandbox/code_execution.

    Returns:
        JSON-serializable dictionary containing:

        {
            "success": bool,
            "status": "success" | "timeout" | "error" | "rejected",
            "returncode": int | None,
            "stdout": str,
            "stderr": str,
            "execution_time": float,
            "timeout": float,
            "truncated": {
                "stdout": bool,
                "stderr": bool
            },
            "blocked_patterns": [str]
        }
    """

    # -----------------------------------------------------------------------
    # Normalize code
    # -----------------------------------------------------------------------

    if code is None:
        code = ""

    if not isinstance(code, str):
        code = str(code)

    # -----------------------------------------------------------------------
    # Validate timeout
    # -----------------------------------------------------------------------

    timeout = _validate_positive_number(
        kwargs.get("timeout"),
        "timeout",
        _DEFAULT_TIMEOUT,
    )

    if timeout is None:
        return _rejected_result(
            "Invalid 'timeout': must be a positive finite number.",
            _DEFAULT_TIMEOUT,
        )

    if timeout > _MAX_TIMEOUT:
        return _rejected_result(
            f"Invalid 'timeout': must not exceed {_MAX_TIMEOUT} seconds.",
            _DEFAULT_TIMEOUT,
        )

    # -----------------------------------------------------------------------
    # Validate code size
    # -----------------------------------------------------------------------

    max_code_size = _validate_positive_integer(
        kwargs.get("max_code_size"),
        "max_code_size",
        _DEFAULT_MAX_CODE_SIZE,
    )

    if max_code_size is None:
        return _rejected_result(
            "Invalid 'max_code_size': must be a positive integer.",
            timeout,
        )

    # -----------------------------------------------------------------------
    # Validate output size
    # -----------------------------------------------------------------------

    max_output_size = _validate_positive_integer(
        kwargs.get("max_output_size"),
        "max_output_size",
        _DEFAULT_MAX_OUTPUT_SIZE,
    )

    if max_output_size is None:
        return _rejected_result(
            "Invalid 'max_output_size': must be a positive integer.",
            timeout,
        )

    # -----------------------------------------------------------------------
    # Validate environment
    # -----------------------------------------------------------------------

    user_env = kwargs.get("env")

    if user_env is not None and not isinstance(user_env, dict):
        return _rejected_result(
            "Invalid 'env': must be a dictionary.",
            timeout,
        )

    # -----------------------------------------------------------------------
    # Validate workspace
    # -----------------------------------------------------------------------

    workspace = _resolve_within_sandbox(
        kwargs.get("workspace", CODE_EXECUTION_ROOT),
        "workspace",
    )

    if workspace is None:
        return _rejected_result(
            "Invalid 'workspace': must be a directory inside "
            "sandbox/code_execution/.",
            timeout,
        )

    # Workspace must exist as a directory.
    try:
        workspace.mkdir(parents=True, exist_ok=True)

        if not workspace.is_dir():
            return _rejected_result(
                "Invalid 'workspace': path is not a directory.",
                timeout,
            )
    except OSError as exc:
        return _rejected_result(
            f"Invalid 'workspace': {exc}",
            timeout,
        )

    # -----------------------------------------------------------------------
    # Validate cwd
    # -----------------------------------------------------------------------

    user_cwd = _resolve_within_sandbox(
        kwargs.get("cwd"),
        "cwd",
    )

    if kwargs.get("cwd") is not None and user_cwd is None:
        return _rejected_result(
            "Invalid 'cwd': must be a directory inside "
            "sandbox/code_execution/.",
            timeout,
        )

    if user_cwd is not None:
        try:
            if not user_cwd.exists() or not user_cwd.is_dir():
                return _rejected_result(
                    "Invalid 'cwd': directory does not exist.",
                    timeout,
                )
        except OSError as exc:
            return _rejected_result(
                f"Invalid 'cwd': {exc}",
                timeout,
            )

    # -----------------------------------------------------------------------
    # Validate code size
    # -----------------------------------------------------------------------

    if len(code) > max_code_size:
        return _rejected_result(
            f"Code exceeds configured maximum size of "
            f"{max_code_size} characters.",
            timeout,
        )

    # -----------------------------------------------------------------------
    # Safety scan
    # -----------------------------------------------------------------------

    blocked_patterns = _looks_dangerous(code)

    if blocked_patterns:
        return _rejected_result(
            "Execution blocked by local safety policy: "
            + ", ".join(blocked_patterns),
            timeout,
            blocked_patterns,
        )

    # -----------------------------------------------------------------------
    # Select working directory
    # -----------------------------------------------------------------------

    working_dir = (
        user_cwd
        if user_cwd is not None
        else workspace
    )

    # -----------------------------------------------------------------------
    # Execute in temporary child process
    # -----------------------------------------------------------------------

    script_path: Optional[Path] = None

    start = time.perf_counter()

    stdout = ""
    stderr = ""
    returncode: Optional[int] = None

    try:
        with TemporaryDirectory(
            prefix="pyexec_",
            dir=str(working_dir),
        ) as tmpdir:

            script_path = Path(tmpdir) / "user_code.py"

            script_path.write_text(
                code,
                encoding="utf-8",
            )

            child_env = _minimal_environment(user_env)

            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(script_path),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(tmpdir),
                    env=child_env,
                    timeout=timeout,
                    check=False,
                )

                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                returncode = completed.returncode

            except subprocess.TimeoutExpired as exc:

                stdout = exc.stdout or ""
                stderr = exc.stderr or ""

                execution_time = (
                    time.perf_counter() - start
                )

                stdout, stdout_truncated = _truncate(
                    stdout,
                    max_output_size,
                )

                stderr, stderr_truncated = _truncate(
                    stderr,
                    max_output_size,
                )

                return {
                    "success": False,
                    "status": "timeout",
                    "returncode": None,
                    "stdout": stdout,
                    "stderr": stderr or "Execution timed out.",
                    "execution_time": round(
                        execution_time,
                        6,
                    ),
                    "timeout": timeout,
                    "truncated": {
                        "stdout": stdout_truncated,
                        "stderr": stderr_truncated,
                    },
                    "blocked_patterns": [],
                }

            # ---------------------------------------------------------------
            # Normal completion
            # ---------------------------------------------------------------

            execution_time = (
                time.perf_counter() - start
            )

            stdout, stdout_truncated = _truncate(
                stdout,
                max_output_size,
            )

            stderr, stderr_truncated = _truncate(
                stderr,
                max_output_size,
            )

            status = (
                "success"
                if returncode == 0
                else "error"
            )

            return {
                "success": returncode == 0,
                "status": status,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": round(
                    execution_time,
                    6,
                ),
                "timeout": timeout,
                "truncated": {
                    "stdout": stdout_truncated,
                    "stderr": stderr_truncated,
                },
                "blocked_patterns": [],
            }

    except Exception as exc:  # pragma: no cover
        execution_time = (
            time.perf_counter() - start
        )

        return {
            "success": False,
            "status": "error",
            "returncode": None,
            "stdout": "",
            "stderr": f"Execution failed: {exc}",
            "execution_time": round(
                execution_time,
                6,
            ),
            "timeout": timeout,
            "truncated": {
                "stdout": False,
                "stderr": False,
            },
            "blocked_patterns": [],
        }

    finally:
        # TemporaryDirectory normally removes this automatically.
        # This is only defensive cleanup.
        if script_path is not None:
            try:
                script_path.unlink(
                    missing_ok=True,
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Executor compatibility wrapper
# ---------------------------------------------------------------------------

def tool(**kwargs: Any) -> Dict[str, Any]:
    """
    Compatibility wrapper for backend.agent.executer.Executor.

    Executor invokes tools approximately as:

        tool(**args)
    """

    code = kwargs.pop("code", "")

    return execute_python_code(
        code,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Direct module test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = execute_python_code(
        "print('hello from code_exec tool')"
    )

    print(result)