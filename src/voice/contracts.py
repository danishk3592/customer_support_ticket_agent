from __future__ import annotations

from abc import ABC, abstractmethod


class STTService(ABC):
    """Contract for speech-to-text adapters."""

    @abstractmethod
    async def transcribe(
        self, audio_bytes: bytes, *, content_type: str | None = None, filename: str | None = None
    ) -> str:
        raise NotImplementedError


class TTSService(ABC):
    """Contract for text-to-speech adapters."""

    media_type: str = "audio/mpeg"

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError
