from __future__ import annotations

import os

from bonsai.adapters.base import AsyncEngineAdapter
from bonsai.adapters.mindie_adapter import MindIEAdapter
from bonsai.adapters.sglang_adapter import SGLangAdapter
from bonsai.adapters.vllm_adapter import VLLMAdapter


def create_engine_adapter() -> AsyncEngineAdapter:
    backend = os.getenv("BONSAI_BACKEND", "vllm").lower()
    model_path = os.getenv("BONSAI_MODEL_PATH")
    if not model_path:
        raise RuntimeError("BONSAI_MODEL_PATH must be set")

    if backend == "sglang":
        return SGLangAdapter(model_path=model_path)
    if backend == "vllm":
        return VLLMAdapter(model_path=model_path)
    if backend == "mindie":
        return MindIEAdapter(model_path=model_path)
    raise RuntimeError(f"Unsupported BONSAI_BACKEND: {backend}")
