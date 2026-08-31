"""Turn router-style chat messages into a Mistral Instruct prompt."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

ChatMessage = Dict[str, str]


def normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[ChatMessage]:
    """Keep only role/content pairs the local model can consume."""
    normalized: List[ChatMessage] = []
    for item in messages:
        role = str(item.get("role") or "user").strip().lower()
        content = item.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        normalized.append({"role": role, "content": text})
    return normalized


def apply_mistral_instruct_template(messages: Sequence[ChatMessage]) -> str:
    """
    Manual Mistral-Instruct chat format used when the tokenizer has no
    `chat_template`. System text is prepended to the first user turn.
    """
    system_parts: List[str] = []
    turns: List[ChatMessage] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            turns.append(message)

    system_prefix = "\n\n".join(system_parts).strip()
    if not turns:
        user_body = system_prefix or "Hello"
        return f"<s>[INST] {user_body} [/INST]"

    chunks: List[str] = ["<s>"]
    first_user_consumed = False
    index = 0
    while index < len(turns):
        turn = turns[index]
        if turn["role"] == "user":
            body = turn["content"]
            if system_prefix and not first_user_consumed:
                body = f"{system_prefix}\n\n{body}"
                first_user_consumed = True
            chunks.append(f"[INST] {body} [/INST]")
            index += 1
            if index < len(turns) and turns[index]["role"] == "assistant":
                chunks.append(f" {turns[index]['content']}</s>")
                index += 1
        else:
            chunks.append(f" {turn['content']}</s>")
            index += 1

    return "".join(chunks)


def render_prompt(messages: Sequence[Dict[str, Any]], tokenizer: Optional[Any] = None) -> str:
    """Render messages with the tokenizer chat template when available."""
    normalized = normalize_messages(messages)
    if not normalized:
        normalized = [{"role": "user", "content": "Hello"}]

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
    return apply_mistral_instruct_template(normalized)
