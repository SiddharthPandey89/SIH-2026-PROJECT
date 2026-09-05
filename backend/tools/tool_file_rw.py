"""
backend/tools/tool_file_rw.py

Local workspace file read/write tool for the Sovereign AI Workbench.

This is a workspace file tool, NOT arbitrary filesystem access.

Allowed paths:
- PROJECT_ROOT/data/            (uploads, documents, outputs, knowledge_base)
- PROJECT_ROOT/sandbox/         (code_execution, temp_outputs)

Security restrictions:
- Paths are resolved before access and must remain inside an allowed root.
- Path traversal (../) and absolute paths escaping the allowed roots are
  rejected.
- No secrets or environment variables are exposed.
- No files or commands are executed.

Supported operations:
- read_file(path, ...)          read a UTF-8 text file
- write_file(path, content, ...) create/overwrite a UTF-8 text file
- tool(**kwargs)                Executor-compatible dispatch wrapper

The executor in backend/agent/executer.py calls tools through a callable API
that expects a function signature compatible with `tool(**args)`. This module
therefore exposes `tool(**kwargs)` as the public entry point used by the agent.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["read_file", "write_file", "tool"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ROOTS: List[Path] = [
    (PROJECT_ROOT / "data").resolve(),
    (PROJECT_ROOT / "sandbox").resolve(),
]

_DEFAULT_MAX_READ_SIZE = 1_000_000  # 1 MB
_DEFAULT_MAX_WRITE_SIZE = 1_000_000  # 1 MB
_DEFAULT_ENCODING = "utf-8"


def _validate_max_size(value: Any, name: str, default: int) -> Optional[int]:
    """
    Validate a size as a strict positive integer.

    Returns the validated integer, or None if the value is invalid.
    Bool, zero, negative, NaN, infinity, non-numeric, and fractional values
    are rejected rather than silently converted.
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


def _rejected_result(reason: str) -> Dict[str, Any]:
    """Build a stable rejected result dictionary."""
    return {
        "success": False,
        "status": "rejected",
        "operation": None,
        "path": None,
        "content": None,
        "error": reason,
    }


def _resolve_within_allowed(path_value: Any) -> Optional[Path]:
    """
    Resolve a path and ensure it stays inside one of the allowed roots.

    Returns the resolved Path if valid, otherwise None.
    """
    if path_value is None:
        return None
    if not isinstance(path_value, (str, Path)):
        return None
    try:
        candidate = Path(path_value).expanduser().resolve()
    except (TypeError, ValueError, OSError):
        return None
    for root in ALLOWED_ROOTS:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return candidate
    return None


def read_file(
    path: Any,
    max_size: Any = None,
    encoding: str = _DEFAULT_ENCODING,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Read a UTF-8 text file from an allowed workspace directory.

    Args:
        path: File path relative to an allowed root, or an absolute path
              inside an allowed root.
        max_size: Maximum file size in bytes allowed to read.
                  Default: 1,000,000 bytes.
        encoding: Text encoding to use. Default: utf-8.

    Returns:
        {
            "success": bool,
            "status": "success" | "error" | "rejected",
            "operation": "read",
            "path": str,
            "content": str | None,
            "error": str | None,
        }
    """
    resolved = _resolve_within_allowed(path)
    if resolved is None:
        return _rejected_result(
            "Invalid 'path': must be inside data/ or sandbox/."
        )

    max_read_size = _validate_max_size(
        max_size, "max_size", _DEFAULT_MAX_READ_SIZE
    )
    if max_read_size is None:
        return _rejected_result(
            "Invalid 'max_size': must be a positive finite number."
        )

    if not isinstance(encoding, str) or not encoding.strip():
        return _rejected_result(
            "Invalid 'encoding': must be a non-empty string."
        )

    try:
        if resolved.is_dir():
            return {
                "success": False,
                "status": "error",
                "operation": "read",
                "path": str(resolved),
                "content": None,
                "error": "Path is a directory, not a file.",
            }

        if not resolved.exists():
            return {
                "success": False,
                "status": "error",
                "operation": "read",
                "path": str(resolved),
                "content": None,
                "error": "File does not exist.",
            }

        file_size = resolved.stat().st_size
        if file_size > max_read_size:
            return {
                "success": False,
                "status": "error",
                "operation": "read",
                "path": str(resolved),
                "content": None,
                "error": (
                    f"File size {file_size} exceeds maximum read size "
                    f"of {max_read_size} bytes."
                ),
            }

        content = resolved.read_text(encoding=encoding)

        return {
            "success": True,
            "status": "success",
            "operation": "read",
            "path": str(resolved),
            "content": content,
            "error": None,
        }

    except PermissionError as exc:
        return {
            "success": False,
            "status": "error",
            "operation": "read",
            "path": str(resolved),
            "content": None,
            "error": f"Permission denied: {exc}",
        }

    except UnicodeDecodeError as exc:
        return {
            "success": False,
            "status": "error",
            "operation": "read",
            "path": str(resolved),
            "content": None,
            "error": f"Encoding error: {exc}",
        }

    except OSError as exc:
        return {
            "success": False,
            "status": "error",
            "operation": "read",
            "path": str(resolved),
            "content": None,
            "error": f"File system error: {exc}",
        }

    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "success": False,
            "status": "error",
            "operation": "read",
            "path": str(resolved),
            "content": None,
            "error": f"Unexpected error: {exc}",
        }


def write_file(
    path: Any,
    content: Any = "",
    mode: str = "write",
    max_size: Any = None,
    encoding: str = _DEFAULT_ENCODING,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Write or append UTF-8 text content to a file in an allowed workspace.

    Args:
        path: File path relative to an allowed root, or an absolute path
              inside an allowed root.
        content: Text content to write. Default: "".
        mode: "write" (overwrite/create) or "append". Default: "write".
        max_size: Maximum resulting file size in bytes.
                  Default: 1,000,000 bytes.
        encoding: Text encoding to use. Default: utf-8.

    Returns:
        {
            "success": bool,
            "status": "success" | "error" | "rejected",
            "operation": "write" | "append",
            "path": str,
            "content": None,
            "error": str | None,
        }
    """
    resolved = _resolve_within_allowed(path)
    if resolved is None:
        return _rejected_result(
            "Invalid 'path': must be inside data/ or sandbox/."
        )

    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)

    if mode not in ("write", "append"):
        return _rejected_result(
            "Invalid 'mode': must be 'write' or 'append'."
        )

    max_write_size = _validate_max_size(
        max_size, "max_size", _DEFAULT_MAX_WRITE_SIZE
    )
    if max_write_size is None:
        return _rejected_result(
            "Invalid 'max_size': must be a positive finite number."
        )

    if not isinstance(encoding, str) or not encoding.strip():
        return _rejected_result(
            "Invalid 'encoding': must be a non-empty string."
        )

    try:
        content_bytes = len(content.encode(encoding))

        if content_bytes > max_write_size:
            return _rejected_result(
                f"Content size {content_bytes} bytes exceeds maximum write "
                f"size of {max_write_size} bytes."
            )

        if mode == "append" and resolved.exists() and resolved.is_file():
            existing_size = resolved.stat().st_size
            resulting_size = existing_size + content_bytes
            if resulting_size > max_write_size:
                return _rejected_result(
                    f"Appending would produce {resulting_size} bytes, "
                    f"exceeding the maximum write size of "
                    f"{max_write_size} bytes."
                )

        if resolved.exists() and resolved.is_dir():
            return {
                "success": False,
                "status": "error",
                "operation": mode,
                "path": str(resolved),
                "content": None,
                "error": "Path is a directory, not a file.",
            }

        resolved.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append":
            with resolved.open("a", encoding=encoding) as handle:
                handle.write(content)
        else:
            with resolved.open("w", encoding=encoding) as handle:
                handle.write(content)

        return {
            "success": True,
            "status": "success",
            "operation": mode,
            "path": str(resolved),
            "content": None,
            "error": None,
        }

    except PermissionError as exc:
        return {
            "success": False,
            "status": "error",
            "operation": mode,
            "path": str(resolved),
            "content": None,
            "error": f"Permission denied: {exc}",
        }

    except UnicodeEncodeError as exc:
        return {
            "success": False,
            "status": "error",
            "operation": mode,
            "path": str(resolved),
            "content": None,
            "error": f"Encoding error: {exc}",
        }

    except LookupError as exc:
        return {
            "success": False,
            "status": "error",
            "operation": mode,
            "path": str(resolved),
            "content": None,
            "error": f"Unknown encoding: {exc}",
        }

    except OSError as exc:
        return {
            "success": False,
            "status": "error",
            "operation": mode,
            "path": str(resolved),
            "content": None,
            "error": f"File system error: {exc}",
        }

    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "success": False,
            "status": "error",
            "operation": mode,
            "path": str(resolved),
            "content": None,
            "error": f"Unexpected error: {exc}",
        }


def tool(**kwargs: Any) -> Dict[str, Any]:
    """
    Executor-compatible dispatch wrapper.

    Supported operations:
        {"operation": "read",  "path": "..."}
        {"operation": "write", "path": "...", "content": "..."}
        {"operation": "append", "path": "...", "content": "..."}

    Also accepts direct keyword forms:
        tool(path="...", content="...", mode="write")
        tool(path="...", mode="read")
    """
    operation = kwargs.get("operation")
    if not isinstance(operation, str):
        operation = ""
    operation = operation.strip().lower()

    if operation == "read":
        return read_file(
            path=kwargs.get("path"),
            max_size=kwargs.get("max_size"),
            encoding=kwargs.get("encoding", _DEFAULT_ENCODING),
        )

    if operation in ("write", "append"):
        return write_file(
            path=kwargs.get("path"),
            content=kwargs.get("content", ""),
            mode=operation,
            max_size=kwargs.get("max_size"),
            encoding=kwargs.get("encoding", _DEFAULT_ENCODING),
        )

    # Direct keyword form: tool(path=..., mode="read")
    mode = kwargs.get("mode")
    if not isinstance(mode, str):
        mode = ""
    mode = mode.strip().lower()

    if mode == "read":
        return read_file(
            path=kwargs.get("path"),
            max_size=kwargs.get("max_size"),
            encoding=kwargs.get("encoding", _DEFAULT_ENCODING),
        )

    if mode in ("write", "append"):
        return write_file(
            path=kwargs.get("path"),
            content=kwargs.get("content", ""),
            mode=mode,
            max_size=kwargs.get("max_size"),
            encoding=kwargs.get("encoding", _DEFAULT_ENCODING),
        )

    return _rejected_result(
        "Invalid 'operation': must be 'read', 'write', or 'append'."
    )


if __name__ == "__main__":
    result = write_file(
        path="sandbox/temp_outputs/hello.txt",
        content="hello from file_rw tool",
    )
    print(result)