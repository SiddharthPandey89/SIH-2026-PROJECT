"""
Discover and load Mistral 7B weights from a local directory or GGUF file.

Supported layouts (all local):
    - Hugging Face transformers directory (config.json + safetensors/bin)
    - A single .gguf file, or a directory containing one or more .gguf files
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

try:
    from .exceptions import MistralGenerationError, MistralModelLoadError, MistralModelPathError
    from .generation import GenerationParams
    from .prompt import render_prompt
    from .settings import MistralSettings
except ImportError:
    from exceptions import MistralGenerationError, MistralModelLoadError, MistralModelPathError
    from generation import GenerationParams
    from prompt import render_prompt
    from settings import MistralSettings

logger = logging.getLogger(__name__)

HF_CONFIG_NAME = "config.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin")
TOKENIZER_HINTS = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")


class LocalInferenceBackend(ABC):
    """In-process generation backend. The router never talks to this directly."""

    name: str
    device: str

    @abstractmethod
    def generate(self, messages: Sequence[dict], params: GenerationParams) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, messages: Sequence[dict], params: GenerationParams) -> Iterator[str]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


def resolve_model_path(path: Path) -> Path:
    if not path.exists():
        raise MistralModelPathError(
            f"Mistral 7B model path does not exist: '{path}'. "
            "Download the weights locally and set MISTRAL_MODEL_PATH."
        )
    return path.resolve()


def detect_model_format(path: Path) -> str:
    """Return 'transformers' or 'llamacpp' based on files on disk."""
    path = resolve_model_path(path)
    if path.is_file():
        if path.suffix.lower() == ".gguf":
            return "llamacpp"
        raise MistralModelPathError(
            f"'{path}' is not a GGUF file or a transformers model directory."
        )

    gguf_files = _find_gguf_files(path)
    has_hf = (path / HF_CONFIG_NAME).is_file() and _has_hf_weights(path)
    if has_hf:
        return "transformers"
    if gguf_files:
        return "llamacpp"
    raise MistralModelPathError(
        f"No usable Mistral 7B weights found under '{path}'. "
        "Expected config.json plus .safetensors/.bin files, or a .gguf file."
    )


def validate_model_files(path: Path, preferred_backend: str = "auto") -> Tuple[str, Path]:
    """
    Confirm the path can be loaded. Returns (backend_name, artifact_path)
    without loading tensors into memory.
    """
    path = resolve_model_path(path)
    detected = detect_model_format(path)
    backend = detected if preferred_backend == "auto" else preferred_backend

    if backend == "transformers":
        if path.is_file():
            raise MistralModelPathError("The transformers backend requires a model directory, not a single file.")
        _assert_hf_layout(path)
        return "transformers", path

    if backend == "llamacpp":
        gguf = path if path.is_file() else _select_gguf(path)
        return "llamacpp", gguf

    raise MistralModelPathError(f"Unsupported Mistral backend '{backend}'.")


def load_backend(settings: MistralSettings) -> LocalInferenceBackend:
    backend_name, artifact = validate_model_files(settings.model_path, settings.preferred_backend)
    logger.info("Loading Mistral 7B via %s from %s", backend_name, artifact)
    if backend_name == "transformers":
        return TransformersBackend.load(settings, artifact)
    return LlamaCppBackend.load(settings, artifact)


def _find_gguf_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.rglob("*.gguf") if p.is_file())


def _has_hf_weights(directory: Path) -> bool:
    if (directory / "model.safetensors.index.json").is_file():
        return True
    if (directory / "pytorch_model.bin.index.json").is_file():
        return True
    for child in directory.iterdir():
        if child.is_file() and child.suffix.lower() in WEIGHT_SUFFIXES:
            return True
    return False


def _assert_hf_layout(directory: Path) -> None:
    missing: List[str] = []
    if not (directory / HF_CONFIG_NAME).is_file():
        missing.append(HF_CONFIG_NAME)
    if not _has_hf_weights(directory):
        missing.append("model weights (.safetensors or .bin)")
    if not any((directory / name).is_file() for name in TOKENIZER_HINTS):
        missing.append("tokenizer files (tokenizer.json / tokenizer.model)")
    if missing:
        raise MistralModelPathError(
            f"Incomplete Hugging Face model directory '{directory}'. Missing: {', '.join(missing)}."
        )


def _select_gguf(directory: Path) -> Path:
    files = _find_gguf_files(directory)
    if not files:
        raise MistralModelPathError(f"No .gguf file found under '{directory}'.")
    preferred = [p for p in files if "instruct" in p.name.lower()]
    pool = preferred or files
    q4 = [p for p in pool if "q4" in p.name.lower()]
    return (q4 or pool)[0]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_torch_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not _cuda_available():
            raise MistralModelLoadError("MISTRAL_DEVICE=cuda but torch.cuda.is_available() is False.")
        return "cuda"
    return "cuda" if _cuda_available() else "cpu"


class TransformersBackend(LocalInferenceBackend):
    name = "transformers"

    def __init__(self, model: object, tokenizer: object, device: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, settings: MistralSettings, model_dir: Path) -> "TransformersBackend":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise MistralModelLoadError(
                "The transformers backend requires 'torch' and 'transformers' to be installed."
            ) from exc

        device = resolve_torch_device(settings.device)
        torch_dtype = _resolve_dtype(settings.dtype, device, torch)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": torch_dtype,
        }
        quantization = _quantization_config(settings, torch)
        if quantization is not None:
            load_kwargs["quantization_config"] = quantization
            load_kwargs["device_map"] = "auto"
        elif device == "cuda":
            load_kwargs["device_map"] = "auto"

        try:
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
            if "device_map" not in load_kwargs:
                model.to(device)
            model.eval()
        except Exception as first_error:
            if device == "cuda" and settings.device == "auto":
                logger.warning(
                    "CUDA load failed (%s); retrying Mistral 7B on CPU.",
                    first_error,
                )
                cpu_kwargs = {
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "torch_dtype": torch.float32,
                }
                try:
                    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **cpu_kwargs)
                    model.to("cpu")
                    model.eval()
                    device = "cpu"
                except Exception as cpu_error:
                    raise MistralModelLoadError(
                        f"Failed to load Mistral 7B from '{model_dir}' on CUDA and CPU: {cpu_error}"
                    ) from cpu_error
            else:
                raise MistralModelLoadError(
                    f"Failed to load Mistral 7B from '{model_dir}': {first_error}"
                ) from first_error

        return cls(model=model, tokenizer=tokenizer, device=device)

    def generate(self, messages: Sequence[dict], params: GenerationParams) -> str:
        pieces = list(self.generate_stream(messages, params))
        return "".join(pieces).strip()

    def generate_stream(self, messages: Sequence[dict], params: GenerationParams) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        prompt = render_prompt(messages, tokenizer=self._tokenizer)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        try:
            target_device = next(self._model.parameters()).device
        except StopIteration:
            target_device = torch.device(self.device)
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = {
            **encoded,
            **params.transformers_kwargs(),
            "streamer": streamer,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        def _run() -> None:
            with torch.inference_mode():
                self._model.generate(**gen_kwargs)

        worker = Thread(target=_run, daemon=True)
        worker.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        worker.join()

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("CUDA cache clear skipped.", exc_info=True)


class LlamaCppBackend(LocalInferenceBackend):
    name = "llamacpp"

    def __init__(self, llm: object, device: str) -> None:
        self._llm = llm
        self.device = device

    @classmethod
    def load(cls, settings: MistralSettings, gguf_path: Path) -> "LlamaCppBackend":
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise MistralModelLoadError(
                "GGUF weights were found but llama-cpp-python is not installed. "
                "Install llama-cpp-python or use a transformers model directory."
            ) from exc

        use_gpu = settings.device != "cpu" and _cuda_available()
        n_gpu_layers = settings.n_gpu_layers if use_gpu else 0
        if n_gpu_layers < 0 and not use_gpu:
            n_gpu_layers = 0

        try:
            llm = Llama(
                model_path=str(gguf_path),
                n_ctx=settings.n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=settings.n_threads,
                chat_format="mistral-instruct",
                verbose=False,
                embedding=False,
            )
        except Exception as first_error:
            if use_gpu and settings.device == "auto":
                logger.warning("llama.cpp GPU load failed (%s); retrying on CPU.", first_error)
                try:
                    llm = Llama(
                        model_path=str(gguf_path),
                        n_ctx=settings.n_ctx,
                        n_gpu_layers=0,
                        n_threads=settings.n_threads,
                        chat_format="mistral-instruct",
                        verbose=False,
                        embedding=False,
                    )
                    use_gpu = False
                except Exception as cpu_error:
                    raise MistralModelLoadError(
                        f"Failed to load GGUF '{gguf_path}' on GPU and CPU: {cpu_error}"
                    ) from cpu_error
            else:
                raise MistralModelLoadError(f"Failed to load GGUF '{gguf_path}': {first_error}") from first_error

        device = "cuda" if use_gpu and n_gpu_layers != 0 else "cpu"
        return cls(llm=llm, device=device)

    def generate(self, messages: Sequence[dict], params: GenerationParams) -> str:
        completion = self._llm.create_chat_completion(
            messages=list(messages),
            stream=False,
            **params.llamacpp_kwargs(),
        )
        try:
            return completion["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise MistralGenerationError("Unexpected llama.cpp chat completion shape.") from exc

    def generate_stream(self, messages: Sequence[dict], params: GenerationParams) -> Iterator[str]:
        stream = self._llm.create_chat_completion(
            messages=list(messages),
            stream=True,
            **params.llamacpp_kwargs(),
        )
        for event in stream:
            try:
                delta = event["choices"][0]["delta"].get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError):
                continue
            if delta:
                yield delta

    def close(self) -> None:
        closer = getattr(self._llm, "close", None)
        if callable(closer):
            closer()
        self._llm = None


def _resolve_dtype(requested: str, device: str, torch_module: object) -> object:
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if requested in mapping:
        return mapping[requested]
    return torch_module.float16 if device == "cuda" else torch_module.float32


def _quantization_config(settings: MistralSettings, torch_module: object) -> Optional[object]:
    if not (settings.load_in_4bit or settings.load_in_8bit):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise MistralModelLoadError(
            "MISTRAL_LOAD_IN_4BIT/8BIT requires bitsandbytes and a BitsAndBytes-enabled transformers build."
        ) from exc
    if settings.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_module.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)
