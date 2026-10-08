from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceTranscriptionResponse(BaseModel):
    success: bool = True
    transcript: str
    processing_time_ms: int = Field(ge=0)


class VoiceSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


class VoiceSynthesisMeta(BaseModel):
    success: bool = True
    processing_time_ms: int = Field(ge=0)
    media_type: str
