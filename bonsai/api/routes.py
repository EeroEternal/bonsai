from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from bonsai.adapters.base import AsyncEngineAdapter
from bonsai.api.schemas import (
    GenerationChunk,
    GenerationParams,
    StatefulGenerateRequest,
    StatefulGenerateResponse,
)
from bonsai.core.errors import EvictedContextError, InvalidIgnoreSpanError
from bonsai.core.state_manager import ContextTreeManager

router = APIRouter()


def get_state_manager(request: Request) -> ContextTreeManager:
    return request.app.state.state_manager


def get_engine_adapter(request: Request) -> AsyncEngineAdapter:
    return request.app.state.engine_adapter


@router.post("/v1/agent/stateful_generate")
async def stateful_generate(
    payload: StatefulGenerateRequest,
    state_manager: ContextTreeManager = Depends(get_state_manager),
    engine_adapter: AsyncEngineAdapter = Depends(get_engine_adapter),
):
    try:
        full_token_ids = state_manager.build_full_token_ids(
            parent_node_id=payload.parent_node_id,
            delta_messages=payload.delta_messages,
            ignore_spans=payload.ignore_spans,
        )
    except (EvictedContextError, InvalidIgnoreSpanError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    params = GenerationParams(
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        stream=payload.stream,
    )

    if payload.stream:
        return StreamingResponse(
            _stream_response(
                payload=payload,
                state_manager=state_manager,
                engine_adapter=engine_adapter,
                full_token_ids=full_token_ids,
                params=params,
            ),
            media_type="application/x-ndjson",
        )

    chunks = [chunk async for chunk in engine_adapter.generate_stream(full_token_ids, params)]
    output_text = "".join(chunk.text for chunk in chunks)
    output_token_ids = _collect_incremental_token_ids(chunks)
    node_id = state_manager.create_child_node(
        session_id=payload.session_id,
        full_token_ids=full_token_ids,
        output_token_ids=output_token_ids,
    )
    return StatefulGenerateResponse(
        session_id=payload.session_id,
        parent_node_id=payload.parent_node_id,
        node_id=node_id,
        output_text=output_text,
        output_token_ids=output_token_ids,
        full_token_ids=full_token_ids,
    )


async def _stream_response(
    *,
    payload: StatefulGenerateRequest,
    state_manager: ContextTreeManager,
    engine_adapter: AsyncEngineAdapter,
    full_token_ids: list[int],
    params: GenerationParams,
) -> AsyncGenerator[bytes, None]:
    emitted_token_ids: list[int] = []
    emitted_text_parts: list[str] = []

    async for chunk in engine_adapter.generate_stream(full_token_ids, params):
        delta_token_ids = _extract_incremental_delta(chunk.token_ids, emitted_token_ids)
        if delta_token_ids:
            emitted_token_ids.extend(delta_token_ids)
        if chunk.text:
            emitted_text_parts.append(chunk.text)

        event = {
            "type": "chunk",
            "delta_text": chunk.text,
            "delta_token_ids": delta_token_ids,
            "finish_reason": chunk.finish_reason,
        }
        yield (json.dumps(event) + "\n").encode("utf-8")

    node_id = state_manager.create_child_node(
        session_id=payload.session_id,
        full_token_ids=full_token_ids,
        output_token_ids=emitted_token_ids,
    )
    done_event = {
        "type": "done",
        "session_id": payload.session_id,
        "parent_node_id": payload.parent_node_id,
        "node_id": node_id,
        "output_text": "".join(emitted_text_parts),
        "output_token_ids": emitted_token_ids,
        "full_token_ids": full_token_ids,
    }
    yield (json.dumps(done_event) + "\n").encode("utf-8")


def _collect_incremental_token_ids(chunks: list[GenerationChunk]) -> list[int]:
    emitted: list[int] = []
    for chunk in chunks:
        emitted.extend(_extract_incremental_delta(chunk.token_ids, emitted))
    return emitted


def _extract_incremental_delta(candidate: list[int], emitted: list[int]) -> list[int]:
    if not candidate:
        return []
    if len(candidate) >= len(emitted) and candidate[: len(emitted)] == emitted:
        return candidate[len(emitted) :]
    return candidate
