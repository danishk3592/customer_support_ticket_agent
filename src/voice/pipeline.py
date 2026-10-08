from __future__ import annotations

import time

from src.utils.errors import AgentProcessingError
from src.voice.contracts import STTService, TTSService
from src.voice.models import VoiceSynthesisMeta, VoiceTranscriptionResponse


class VoicePipeline:
    """Shared adapter boundary; never calls the text-agent workflow directly.

    STT produces text for the existing POST /chat flow. TTS accepts the exact
    displayed assistant text and returns playable bytes.
    """

    def __init__(self, stt_service: STTService, tts_service: TTSService) -> None:
        self.stt = stt_service
        self.tts = tts_service

    async def transcribe(
        self, audio_bytes: bytes, *, content_type: str | None = None, filename: str | None = None
    ) -> VoiceTranscriptionResponse:
        if not audio_bytes:
            raise ValueError("Audio input is empty")
        started = time.perf_counter()
        try:
            # Whisper inference is CPU/GPU-bound; keep it off FastAPI's event loop.
            transcript = await self.stt.transcribe(
                audio_bytes, content_type=content_type, filename=filename
            )
        except Exception as exc:
            raise AgentProcessingError("Speech transcription failed") from exc
        processing_time_ms = max(0, round((time.perf_counter() - started) * 1000))
        clean = (transcript or "").strip()
        if not clean:
            raise AgentProcessingError("No intelligible speech was detected")
        return VoiceTranscriptionResponse(
            transcript=clean, processing_time_ms=processing_time_ms
        )

    async def synthesize(self, text: str) -> tuple[bytes, VoiceSynthesisMeta]:
        clean = text.strip()
        if not clean:
            raise ValueError("Response text must not be blank")
        started = time.perf_counter()
        try:
            audio = await self.tts.synthesize(clean)
        except Exception as exc:
            raise AgentProcessingError("Speech synthesis failed") from exc
        processing_time_ms = max(0, round((time.perf_counter() - started) * 1000))
        if not audio:
            raise AgentProcessingError("Speech synthesis returned empty audio")
        meta = VoiceSynthesisMeta(
            processing_time_ms=processing_time_ms,
            media_type=self.tts.media_type,
        )
        return audio, meta


