from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import uuid4

from bonsai.adapters.base import AsyncEngineAdapter
from bonsai.api.schemas import GenerationChunk, GenerationParams


class VLLMAdapter(AsyncEngineAdapter):
    def __init__(self, model_path: str, **engine_kwargs):
        try:
            from vllm import SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
        except ImportError as exc:
            raise RuntimeError("vllm is required for the vllm backend") from exc

        engine_args = AsyncEngineArgs(
            model=model_path,
            enable_prefix_caching=True,
            **engine_kwargs,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params_cls = SamplingParams

    async def generate_stream(
        self,
        token_ids: list[int],
        params: GenerationParams,
    ) -> AsyncGenerator[GenerationChunk, None]:
        sampling_params = self.sampling_params_cls(
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
        )
        generator = self.engine.generate(
            prompt=None,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            request_id=uuid4().hex,
        )
        async for chunk in generator:
            yield self._coerce_chunk(chunk)

    def _coerce_chunk(self, chunk) -> GenerationChunk:
        outputs = getattr(chunk, "outputs", None)
        if outputs:
            output = outputs[-1]
            return GenerationChunk(
                text=getattr(output, "text", ""),
                token_ids=list(getattr(output, "token_ids", []) or []),
                finish_reason=getattr(output, "finish_reason", None),
            )
        return GenerationChunk()
