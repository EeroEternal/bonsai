from __future__ import annotations

import os

from fastapi import FastAPI

from bonsai.adapters.factory import create_engine_adapter
from bonsai.api.routes import router
from bonsai.core.state_manager import ContextTreeManager


def create_app(
    *,
    state_manager: ContextTreeManager | None = None,
    engine_adapter=None,
) -> FastAPI:
    app = FastAPI(title="Project Bonsai")
    tokenizer_model = os.getenv("BONSAI_TOKENIZER_MODEL")

    if state_manager is None:
        if not tokenizer_model:
            raise RuntimeError("BONSAI_TOKENIZER_MODEL must be set")
        state_manager = ContextTreeManager.from_pretrained(tokenizer_model)

    if engine_adapter is None:
        engine_adapter = create_engine_adapter()

    app.state.state_manager = state_manager
    app.state.engine_adapter = engine_adapter
    app.include_router(router)
    return app


if os.getenv("BONSAI_TOKENIZER_MODEL") and os.getenv("BONSAI_MODEL_PATH"):
    app = create_app()
else:
    app = FastAPI(title="Project Bonsai")
