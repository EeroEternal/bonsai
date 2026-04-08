import pytest

from bonsai.api.schemas import ChatMessage
from bonsai.core.errors import EvictedContextError, InvalidIgnoreSpanError
from bonsai.core.state_manager import ContextTreeManager


class FakeTokenizer:
    bos_token_id = 101

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        token_ids = [self.bos_token_id]
        for message in messages:
            token_ids.append(len(message["role"]))
            token_ids.extend(ord(ch) for ch in message["content"])
        token_ids.append(999)
        return token_ids


def test_build_full_token_ids_prunes_parent_and_strips_duplicate_bos():
    manager = ContextTreeManager(tokenizer=FakeTokenizer(), max_nodes=4)
    manager.cache["parent"] = [1, 2, 3, 4, 5]

    full_ids = manager.build_full_token_ids(
        parent_node_id="parent",
        delta_messages=[ChatMessage(role="user", content="go")],
        ignore_spans=[(1, 3)],
    )

    assert full_ids == [1, 4, 5, 4, 103, 111, 999]


def test_build_full_token_ids_rejects_missing_parent():
    manager = ContextTreeManager(tokenizer=FakeTokenizer(), max_nodes=4)

    with pytest.raises(EvictedContextError):
        manager.build_full_token_ids(
            parent_node_id="missing",
            delta_messages=[ChatMessage(role="user", content="x")],
        )


def test_build_full_token_ids_rejects_invalid_ignore_span():
    manager = ContextTreeManager(tokenizer=FakeTokenizer(), max_nodes=4)
    manager.cache["parent"] = [1, 2, 3]

    with pytest.raises(InvalidIgnoreSpanError):
        manager.build_full_token_ids(
            parent_node_id="parent",
            delta_messages=[ChatMessage(role="user", content="x")],
            ignore_spans=[(2, 2)],
        )
