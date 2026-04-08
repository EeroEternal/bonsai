from __future__ import annotations

from hashlib import sha256
from typing import Iterable, Sequence

from cachetools import LRUCache
from transformers import AutoTokenizer

from bonsai.api.schemas import ChatMessage
from bonsai.core.errors import EvictedContextError, InvalidIgnoreSpanError


class ContextTreeManager:
    def __init__(self, tokenizer, max_nodes: int = 1024):
        self.tokenizer = tokenizer
        self.cache: LRUCache[str, list[int]] = LRUCache(maxsize=max_nodes)

    @classmethod
    def from_pretrained(cls, model_name: str, max_nodes: int = 1024) -> "ContextTreeManager":
        tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
        return cls(tokenizer=tokenizer, max_nodes=max_nodes)

    def build_full_token_ids(
        self,
        *,
        parent_node_id: str | None,
        delta_messages: Sequence[ChatMessage],
        ignore_spans: Sequence[tuple[int, int]] | None = None,
    ) -> list[int]:
        parent_ids = self._get_parent_ids(parent_node_id)
        pruned_parent_ids = self._apply_ignore_spans(parent_ids, ignore_spans)
        delta_ids = self._tokenize_delta_messages(delta_messages, has_parent=parent_node_id is not None)
        return pruned_parent_ids + delta_ids

    def create_child_node(self, *, session_id: str, full_token_ids: Sequence[int], output_token_ids: Sequence[int]) -> str:
        final_ids = list(full_token_ids) + list(output_token_ids)
        node_id = sha256(f"{session_id}:{','.join(map(str, final_ids))}".encode("utf-8")).hexdigest()[:24]
        self.cache[node_id] = final_ids
        return node_id

    def get_node(self, node_id: str) -> list[int]:
        try:
            return list(self.cache[node_id])
        except KeyError as exc:
            raise EvictedContextError(f"Unknown or evicted node_id: {node_id}") from exc

    def _get_parent_ids(self, parent_node_id: str | None) -> list[int]:
        if parent_node_id is None:
            return []
        return self.get_node(parent_node_id)

    def _apply_ignore_spans(
        self,
        token_ids: Sequence[int],
        ignore_spans: Sequence[tuple[int, int]] | None,
    ) -> list[int]:
        if not ignore_spans:
            return list(token_ids)

        normalized = self._normalize_spans(ignore_spans, len(token_ids))
        pruned_ids: list[int] = []
        cursor = 0
        for start, end in normalized:
            pruned_ids.extend(token_ids[cursor:start])
            cursor = end
        pruned_ids.extend(token_ids[cursor:])
        return pruned_ids

    def _normalize_spans(
        self,
        spans: Sequence[tuple[int, int]],
        token_count: int,
    ) -> list[tuple[int, int]]:
        ordered = sorted(spans)
        merged: list[tuple[int, int]] = []
        for start, end in ordered:
            if start < 0 or end < 0 or start >= end:
                raise InvalidIgnoreSpanError(
                    f"Invalid ignore span {start, end}: spans must be non-negative and non-empty"
                )
            if end > token_count:
                raise InvalidIgnoreSpanError(f"Ignore span exceeds token count: {(start, end)} > {token_count}")
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
                continue
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def _tokenize_delta_messages(self, delta_messages: Sequence[ChatMessage], *, has_parent: bool) -> list[int]:
        message_dicts = [message.model_dump() for message in delta_messages]
        token_ids = list(
            self.tokenizer.apply_chat_template(
                message_dicts,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if has_parent and bos_token_id is not None and token_ids and token_ids[0] == bos_token_id:
            return token_ids[1:]
        return token_ids
