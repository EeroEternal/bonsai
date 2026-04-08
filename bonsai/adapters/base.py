from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from bonsai.api.schemas import GenerationChunk, GenerationParams


class AsyncEngineAdapter(ABC):
    @abstractmethod
    async def generate_stream(
        self,
        token_ids: list[int],
        params: GenerationParams,
    ) -> AsyncGenerator[GenerationChunk, None]:
        raise NotImplementedError
