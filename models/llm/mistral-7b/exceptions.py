"""Errors raised by the local Mistral 7B engine."""

from __future__ import annotations


class MistralError(Exception):
    """Base exception for the local Mistral 7B component."""


class MistralConfigError(MistralError):
    """Invalid configuration (device, generation parameters, environment)."""


class MistralModelPathError(MistralError):
    """The configured weights path is missing, empty, or not a usable model tree."""


class MistralModelLoadError(MistralError):
    """Weights were found but could not be loaded into memory."""


class MistralGenerationError(MistralError):
    """Inference failed after a successful load."""


class MistralNotLoadedError(MistralError):
    """An operation required a loaded model, but the engine has been shut down."""
