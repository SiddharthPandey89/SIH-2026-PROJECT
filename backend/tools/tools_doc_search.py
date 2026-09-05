"""Executor-facing adapter for local knowledge-base document search."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from backend.knowledge_base.retriever import get_retriever

__all__ = ["doc_search_tool", "tool"]

logger = logging.getLogger(__name__)

_SUPPORTED_OPERATIONS = {"search", "health"}
_DEFAULT_TOP_K = 5
_MAX_TOP_K = 20
_MAX_QUERY_LENGTH = 8_000
_MAX_FILE_ID_LENGTH = 256


def _result(
	success: bool,
	status: str,
	operation: Optional[str],
	message: str,
	error: Optional[str] = None,
	data: Any = None,
) -> Dict[str, Any]:
	"""Build the stable, JSON-serializable tool result contract."""
	return {
		"success": success,
		"status": status,
		"operation": operation,
		"message": message,
		"error": error,
		"data": data,
	}


def _reject(operation: Optional[str], message: str) -> Dict[str, Any]:
	return _result(False, "rejected", operation, message)


def _validate_search_inputs(
	query: Any,
	file_id: Any,
	top_k: Any,
) -> tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
	"""Validate and normalize search arguments without changing their meaning."""
	if not isinstance(query, str):
		return None, None, None, "query must be a string."
	normalized_query = query.strip()
	if not normalized_query:
		return None, None, None, "query must not be empty or whitespace-only."
	if len(normalized_query) > _MAX_QUERY_LENGTH:
		return None, None, None, f"query must not exceed {_MAX_QUERY_LENGTH} characters."

	normalized_file_id: Optional[str] = None
	if file_id is not None:
		if not isinstance(file_id, str):
			return None, None, None, "file_id must be a string when supplied."
		normalized_file_id = file_id.strip()
		if not normalized_file_id:
			return None, None, None, "file_id must not be empty or whitespace-only."
		if len(normalized_file_id) > _MAX_FILE_ID_LENGTH:
			return None, None, None, f"file_id must not exceed {_MAX_FILE_ID_LENGTH} characters."

	if top_k is None:
		normalized_top_k = _DEFAULT_TOP_K
	elif isinstance(top_k, bool) or not isinstance(top_k, int):
		return None, None, None, "top_k must be an integer."
	else:
		normalized_top_k = top_k

	if normalized_top_k < 1 or normalized_top_k > _MAX_TOP_K:
		return None, None, None, f"top_k must be between 1 and {_MAX_TOP_K}."

	return normalized_query, normalized_file_id, normalized_top_k, None


def _normalize_search_results(results: Any) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
	"""Validate Retriever records and convert them to plain JSON values."""
	if not isinstance(results, (list, tuple)):
		return None, "Retriever returned an invalid result collection."

	normalized: List[Dict[str, Any]] = []
	for index, item in enumerate(results):
		if not isinstance(item, Mapping):
			return None, f"Retriever returned an invalid result at index {index}."

		required_fields = ("document_id", "title", "snippet", "score")
		if any(field not in item for field in required_fields):
			return None, f"Retriever result at index {index} is missing required fields."

		document_id = item["document_id"]
		title = item["title"]
		snippet = item["snippet"]
		score = item["score"]

		if not isinstance(document_id, str) or not isinstance(title, str) or not isinstance(snippet, str):
			return None, f"Retriever result at index {index} contains invalid text fields."
		if isinstance(score, bool) or not isinstance(score, (int, float)):
			return None, f"Retriever result at index {index} contains an invalid score."
		if not math.isfinite(float(score)):
			return None, f"Retriever result at index {index} contains a non-finite score."

		normalized.append({
			"document_id": document_id,
			"title": title,
			"snippet": snippet,
			"score": float(score),
		})

	return normalized, None


async def doc_search_tool(
	operation: Any,
	query: Any = None,
	file_id: Any = None,
	top_k: Any = None,
) -> Dict[str, Any]:
	"""Search the local Retriever or report its health.

	The function is asynchronous because the repository Retriever exposes
	asynchronous methods. The Executor detects and awaits the returned
	coroutine, so this adapter does not create or manage event loops.
	"""
	if not isinstance(operation, str):
		return _reject(None, "operation must be a string.")
	normalized_operation = operation.strip().lower()
	if normalized_operation not in _SUPPORTED_OPERATIONS:
		return _reject(normalized_operation, "operation must be 'search' or 'health'.")

	if normalized_operation == "health":
		if query is not None or file_id is not None or top_k is not None:
			return _reject("health", "health does not accept query, file_id, or top_k.")
		try:
			retriever = get_retriever()
			healthy = await retriever.health_check()
		except Exception as exc:
			logger.exception("Document search Retriever health check failed.")
			return _result(
				False,
				"error",
				"health",
				"Document search health check failed.",
				f"Internal health-check error: {type(exc).__name__}.",
				{"healthy": False},
			)
		if not isinstance(healthy, bool):
			return _result(
				False,
				"error",
				"health",
				"Retriever returned an invalid health status.",
				"Health status must be boolean.",
				{"healthy": False},
			)
		if not healthy:
			return _result(
				False,
				"error",
				"health",
				"Document search Retriever is unavailable.",
				"Retriever health check reported unavailable.",
				{"healthy": False},
			)
		return _result(True, "success", "health", "Document search Retriever is healthy.", data={"healthy": True})

	normalized_query, normalized_file_id, normalized_top_k, validation_error = _validate_search_inputs(
		query,
		file_id,
		top_k,
	)
	if validation_error:
		return _reject("search", validation_error)

	try:
		retriever = get_retriever()
		raw_results = await retriever.retrieve_context(
			query=normalized_query,
			file_id=normalized_file_id,
			top_k=normalized_top_k,
		)
		results, normalization_error = _normalize_search_results(raw_results)
		if normalization_error:
			return _result(
				False,
				"error",
				"search",
				"Retriever returned malformed search results.",
				normalization_error,
			)
	except Exception as exc:
		logger.exception("Document search Retriever query failed.")
		return _result(
			False,
			"error",
			"search",
			"Document search failed.",
			f"Internal search error: {type(exc).__name__}.",
		)

	if not results:
		message = "No relevant documents found."
	else:
		message = "Document search completed successfully."
	return _result(
		True,
		"success",
		"search",
		message,
		data={
			"query": normalized_query,
			"file_id": normalized_file_id,
			"top_k": normalized_top_k,
			"results": results,
			"count": len(results),
		},
	)


async def tool(**kwargs: Any) -> Dict[str, Any]:
	"""Executor-compatible asynchronous wrapper."""
	allowed = {"operation", "query", "file_id", "top_k"}
	unexpected = sorted(set(kwargs) - allowed)
	if unexpected:
		return _reject(None, "Unexpected keyword argument(s): " + ", ".join(unexpected) + ".")
	if "operation" not in kwargs:
		return _reject(None, "operation is required.")
	return await doc_search_tool(**kwargs)
