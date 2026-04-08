from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class StatefulGenerateRequest(BaseModel):
    session_id: str = Field(min_length=1)
    parent_node_id: str | None = None
    delta_messages: list[ChatMessage] = Field(min_length=1)
    ignore_spans: list[tuple[int, int]] | None = None
    max_new_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0)
    stream: bool = True


class GenerationParams(BaseModel):
    max_new_tokens: int
    temperature: float
    stream: bool


class GenerationChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = ""
    token_ids: list[int] = Field(default_factory=list)
    finish_reason: str | None = None


class StatefulGenerateResponse(BaseModel):
    session_id: str
    parent_node_id: str | None = None
    node_id: str
    output_text: str
    output_token_ids: list[int]
    full_token_ids: list[int]
