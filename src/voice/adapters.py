from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from src.config import Settings
from src.voice.contracts import STTService, TTSService


class FasterWhisperSTTService(STTService):
    """Local Whisper adapter using faster-whisper; the model loads lazily."""

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.stt_model
        self.device = settings.stt_device
        self.compute_type = settings.stt_compute_type
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install project requirements first."
                ) from exc
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def _transcribe_sync(self, audio_bytes: bytes, filename: str | None) -> str:
        suffix = Path(filename or "audio.wav").suffix.lower() or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
            temp.write(audio_bytes)
            temp.flush()
            model = self._get_model()
            segments, _ = model.transcribe(
                temp.name,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

    async def transcribe(
        self, audio_bytes: bytes, *, content_type: str | None = None, filename: str | None = None
    ) -> str:
        if not audio_bytes:
            raise ValueError("Audio input is empty")
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, filename)


class EdgeTTSService(TTSService):
    """Edge TTS adapter returning MP3 bytes without an API key."""

    media_type = "audio/mpeg"

    def __init__(self, settings: Settings) -> None:
        self.voice = settings.tts_voice

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("Response text must not be blank")
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Install project requirements first."
            ) from exc

        communicator = edge_tts.Communicate(text.strip(), self.voice)
        chunks: list[bytes] = []
        async for chunk in communicator.stream():
            if chunk.get("type") == "audio":
                data = chunk.get("data") or b""
                if data:
                    chunks.append(data)
        return b"".join(chunks)
