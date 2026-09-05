"""
backend/tools/tool_output_gen.py

Local/offline output generation adapter for the Sovereign AI Workbench.

This module is intentionally thin: it delegates to the existing writer modules
that already enforce workspace security, validation, overwrite rules, and atomic
writes.

Public API:
-----------
    generate_output(format, content, path, overwrite=False, **kwargs)
    tool(**kwargs)

The tool is designed to be callable from the agent/executor layer while
remaining completely local and offline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from backend.output_gen.docx_writer import ALLOWED_ROOTS, write_docx
from backend.output_gen.pptx_writer import write_pptx
from backend.output_gen.xlsx_writer import write_xlsx

__all__ = ["generate_output", "tool"]

logger = logging.getLogger(__name__)

_FORMAT_MAP = {
    "docx": write_docx,
    "pptx": write_pptx,
    "xlsx": write_xlsx,
}

_FORMAT_ALIASES = {
    "docx": "docx",
    ".docx": "docx",
    "pptx": "pptx",
    ".pptx": "pptx",
    "xlsx": "xlsx",
    ".xlsx": "xlsx",
}


def _normalize_format(format_value: Any) -> Optional[str]:
    """Normalize a user-supplied format value to a supported lowercase key."""
    if format_value is None:
        return None
    if not isinstance(format_value, str):
        return None
    value = format_value.strip().lower()
    if not value:
        return None
    return _FORMAT_ALIASES.get(value)


def _result(
    success: bool,
    status: str,
    format_name: Optional[str],
    path: Optional[str],
    message: Optional[str],
    error: Optional[str],
) -> Dict[str, Any]:
    """Build the adapter's exact six-key result contract."""
    return {
        "success": success,
        "status": status,
        "format": format_name,
        "path": path,
        "message": message,
        "error": error,
    }


def _normalize_result(
    result: Any,
    format_name: Optional[str],
    requested_path: Optional[str],
) -> Dict[str, Any]:
    """Normalize writer output into a stable tool-level result contract."""
    if not isinstance(result, Mapping):
        return _result(False, "error", format_name, requested_path,
                       "Writer returned an invalid result object.",
                       "Writer returned an invalid result object.")

    success = bool(result.get("success", False))
    status = result.get("status")
    if status is None:
        status = "success" if success else ("rejected" if not result.get("path") else "error")
    if status not in {"success", "rejected", "error"}:
        return _result(False, "error", format_name, requested_path,
                       "Writer returned an invalid status.",
                       "Writer returned an invalid status.")

    format_value = result.get("format") or format_name
    path_value = result.get("path") or requested_path
    message = result.get("message")
    error_value = result.get("error")

    if success and status != "success":
        return _result(False, "error", format_value, requested_path,
                       "Writer returned an inconsistent result.",
                       "Writer reported success with a non-success status.")

    if success:
        return _result(True, "success", format_value,
                       str(path_value) if path_value is not None else None,
                       str(message) if message else "Output generated successfully.",
                       None)

    if status == "rejected":
        return _result(False, "rejected", format_value, None,
                       str(message or error_value or "Request was rejected by validation."),
                       None)

    return _result(False, "error", format_value,
                   str(path_value) if path_value is not None else None,
                   str(message) if message else "Output generation failed.",
                   str(error_value) if error_value is not None else "Output generation failed.")


def _validate_path_and_format(format_name: str, path_value: Any) -> Path:
    """Validate path and extension at the tool layer before dispatch."""
    if path_value is None:
        raise ValueError("A file path is required.")
    if not isinstance(path_value, (str, Path)):
        raise TypeError("Path must be a string or pathlib.Path object.")

    try:
        candidate = Path(path_value).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("Path is invalid.") from exc

    if not candidate.name:
        raise ValueError("Path must include a filename.")

    if not any(candidate == root or root in candidate.parents for root in ALLOWED_ROOTS):
        raise ValueError("Path must be inside the project's data/ or sandbox/ directory.")

    expected_suffix = f".{format_name}"
    actual_suffix = candidate.suffix.lower()
    if actual_suffix and actual_suffix != expected_suffix:
        raise ValueError(
            f"Format '{format_name}' requires a '{expected_suffix}' extension, "
            f"got '{actual_suffix}' instead."
        )

    if not actual_suffix:
        raise ValueError(
            f"Format '{format_name}' requires a '{expected_suffix}' extension."
        )

    if candidate.is_dir():
        raise ValueError("Path points to a directory, not a file.")

    return candidate


def generate_output(
    format: Any,
    content: Any,
    path: Any,
    overwrite: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate a local file in the supported output formats.

    This module does not implement generation logic itself; it delegates to the
    repository's existing writer modules.

    Args:
        format: Output format such as "docx", "pptx", or "xlsx".
        content: Structured content for the chosen writer.
        path: Destination path for the generated file.
        overwrite: Whether to replace an existing file when allowed.
        **kwargs: Extra writer options forwarded without alteration.

    Returns:
        {
            "success": bool,
            "status": "success" | "rejected" | "error",
            "format": str | None,
            "path": str | None,
            "message": str | None,
            "error": str | None,
        }
    """
    requested_format = _normalize_format(format)
    if requested_format is None:
        return _result(False, "rejected", None, None,
                       "Unsupported or missing output format. Use docx, pptx, or xlsx.",
                       None)

    if requested_format not in _FORMAT_MAP:
        return _result(False, "rejected", requested_format, None,
                       "Unsupported output format.", None)

    if not isinstance(content, Mapping):
        return _result(False, "rejected", requested_format, None,
                       "Content must be a mapping.", None)

    if not isinstance(overwrite, bool):
        return _result(False, "rejected", requested_format, None,
                       "overwrite must be a boolean value.", None)

    if kwargs:
        return _result(False, "rejected", requested_format, None,
                       "Unsupported writer options were supplied.", None)

    try:
        validated_path = _validate_path_and_format(requested_format, path)
    except (TypeError, ValueError) as exc:
        return _result(False, "rejected", requested_format, None, str(exc), None)

    try:
        writer = _FORMAT_MAP[requested_format]
        result = writer(dict(content), str(validated_path), overwrite=overwrite)
        normalized = _normalize_result(result, requested_format, str(validated_path))

        if normalized["success"]:
            returned_path = normalized.get("path")
            try:
                final_path = _validate_path_and_format(requested_format, returned_path)
            except (TypeError, ValueError):
                return _result(False, "error", requested_format, str(validated_path),
                               "Writer reported success with an invalid output path.",
                               "Writer returned an invalid output path.")
            if not final_path.is_file() or final_path.stat().st_size == 0:
                return _result(False, "error", requested_format, str(final_path),
                               "Writer reported success but the output file is missing or empty.",
                               "Generated output file was not created correctly.")

        logger.info(
            "Output generation request completed",
            extra={
                "operation": "generate_output",
                "format": requested_format,
                "path": str(validated_path),
                "success": normalized["success"],
            },
        )
        return normalized

    except PermissionError as exc:
        return _result(False, "error", requested_format, str(validated_path),
                       "Permission denied.", str(exc))
    except OSError as exc:
        return _result(False, "error", requested_format, str(validated_path),
                       "Filesystem error.", str(exc))
    except Exception as exc:  # Writers may raise library-specific generation errors.
        logger.exception("Unexpected error in output generation tool")
        return _result(False, "error", requested_format, str(validated_path),
                       "Unexpected output generation error.",
                       f"Unexpected error: {type(exc).__name__}")


def tool(**kwargs: Any) -> Dict[str, Any]:
    """Executor-compatible wrapper for output generation tools."""
    return generate_output(**kwargs)
