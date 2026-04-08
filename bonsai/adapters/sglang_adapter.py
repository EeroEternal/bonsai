from __future__ import annotations

from collections.abc import AsyncGenerator

from bonsai.adapters.base import AsyncEngineAdapter
from bonsai.api.schemas import GenerationChunk, GenerationParams


class SGLangAdapter(AsyncEngineAdapter):
    def __init__(self, model_path: str, **engine_kwargs):
        try:
            import sglang as sgl
        except ImportError as exc:
            raise RuntimeError("sglang is required for the sglang backend") from exc

        self.engine = sgl.Engine(model_path=model_path, **engine_kwargs)

    async def generate_stream(
        self,
        token_ids: list[int],
        params: GenerationParams,
    ) -> AsyncGenerator[GenerationChunk, None]:
        result = await self.engine.async_generate(
            {"input_ids": token_ids},
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            stream=params.stream,
        )

        if hasattr(result, "__aiter__"):
            async for chunk in result:
                yield self._coerce_chunk(chunk)
            return

        yield self._coerce_chunk(result)

    def _coerce_chunk(self, chunk) -> GenerationChunk:
        return GenerationChunk(
            text=getattr(chunk, "text", "") or chunk.get("text", ""),
            token_ids=list(getattr(chunk, "token_ids", []) or chunk.get("token_ids", [])),
            finish_reason=getattr(chunk, "finish_reason", None) or chunk.get("finish_reason"),
        )
