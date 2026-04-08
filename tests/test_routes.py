import json

from fastapi.testclient import TestClient

from bonsai.adapters.base import AsyncEngineAdapter
from bonsai.api.schemas import GenerationChunk, GenerationParams
from bonsai.core.state_manager import ContextTreeManager
from bonsai.main import create_app


class FakeTokenizer:
    bos_token_id = 11

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [self.bos_token_id, 21, len(messages[0]["content"]), 99]


class FakeEngineAdapter(AsyncEngineAdapter):
    async def generate_stream(self, token_ids: list[int], params: GenerationParams):
        assert params.max_new_tokens == 16
        assert params.temperature == 0.0
        yield GenerationChunk(text="Hel", token_ids=[7])
        yield GenerationChunk(text="lo", token_ids=[7, 8], finish_reason="stop")


def test_stateful_generate_json_response():
    app = create_app(
        state_manager=ContextTreeManager(tokenizer=FakeTokenizer(), max_nodes=4),
        engine_adapter=FakeEngineAdapter(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/agent/stateful_generate",
        json={
            "session_id": "session-1",
            "delta_messages": [{"role": "user", "content": "ping"}],
            "max_new_tokens": 16,
            "temperature": 0.0,
            "stream": False,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["session_id"] == "session-1"
    assert body["output_text"] == "Hello"
    assert body["output_token_ids"] == [7, 8]
    assert body["full_token_ids"] == [11, 21, 4, 99]
    assert body["node_id"]


def test_stateful_generate_streaming_response():
    app = create_app(
        state_manager=ContextTreeManager(tokenizer=FakeTokenizer(), max_nodes=4),
        engine_adapter=FakeEngineAdapter(),
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/agent/stateful_generate",
        json={
            "session_id": "session-2",
            "delta_messages": [{"role": "user", "content": "ping"}],
            "max_new_tokens": 16,
            "temperature": 0.0,
            "stream": True,
        },
    ) as response:
        lines = [json.loads(line) for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert lines[0]["type"] == "chunk"
    assert lines[0]["delta_token_ids"] == [7]
    assert lines[1]["delta_token_ids"] == [8]
    assert lines[-1]["type"] == "done"
    assert lines[-1]["output_text"] == "Hello"
