from __future__ import annotations

from collections.abc import AsyncGenerator

from bonsai.adapters.base import AsyncEngineAdapter
from bonsai.adapters.vllm_adapter import VLLMAdapter
from bonsai.api.schemas import GenerationChunk, GenerationParams


class MindIEAdapter(AsyncEngineAdapter):
    def __init__(self, model_path: str, **engine_kwargs):
        self._delegate: AsyncEngineAdapter | None = None
        try:
            self._delegate = VLLMAdapter(model_path=model_path, device="npu", **engine_kwargs)
        except RuntimeError:
            self._delegate = None

        self.model_path = model_path
        self.engine_kwargs = engine_kwargs

    async def generate_stream(
        self,
        token_ids: list[int],
        params: GenerationParams,
    ) -> AsyncGenerator[GenerationChunk, None]:
        if self._delegate is not None:
            async for chunk in self._delegate.generate_stream(token_ids, params):
                yield chunk
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the native mindie backend") from exc

        try:
            import mindie_llm
        except ImportError as exc:
            raise RuntimeError("mindie_llm or vllm-ascend is required for the mindie backend") from exc

        input_ids = torch.tensor([token_ids], dtype=torch.int64, device="npu")
        stream = await mindie_llm.async_generate(
            input_ids=input_ids,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            stream=params.stream,
            model_path=self.model_path,
            **self.engine_kwargs,
        )
        async for chunk in stream:
            yield GenerationChunk(
                text=getattr(chunk, "text", ""),
                token_ids=list(getattr(chunk, "token_ids", []) or []),
                finish_reason=getattr(chunk, "finish_reason", None),
            )
