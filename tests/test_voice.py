import pytest

from src.voice.pipeline import VoicePipeline
from src.voice.playback import VoicePlaybackStore
from src.utils.errors import AgentProcessingError


class FakeSTT:
    async def transcribe(self, audio_bytes: bytes, *, content_type=None, filename=None) -> str:
        assert audio_bytes
        return "My payment was charged twice"


class FailingSTT:
    async def transcribe(self, audio_bytes: bytes, *, content_type=None, filename=None) -> str:
        raise RuntimeError("decoder failure")


class FakeTTS:
    media_type = "audio/mpeg"

    async def synthesize(self, text: str) -> bytes:
        return ("AUDIO:" + text).encode()


class FailingTTS:
    media_type = "audio/mpeg"

    async def synthesize(self, text: str) -> bytes:
        raise RuntimeError("provider failure")


@pytest.mark.asyncio
async def test_successful_transcription_and_timing():
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    result = await pipeline.transcribe(b"fake-audio", content_type="audio/wav", filename="voice.wav")
    assert result.transcript == "My payment was charged twice"
    assert result.processing_time_ms >= 0


@pytest.mark.asyncio
async def test_empty_audio_validation():
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    with pytest.raises(ValueError, match="empty"):
        await pipeline.transcribe(b"")


@pytest.mark.asyncio
async def test_stt_failure_handling():
    pipeline = VoicePipeline(FailingSTT(), FakeTTS())
    with pytest.raises(AgentProcessingError, match="transcription failed"):
        await pipeline.transcribe(b"fake-audio")


@pytest.mark.asyncio
async def test_successful_synthesis_and_media_type():
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    audio, meta = await pipeline.synthesize("Hello customer")
    assert audio == b"AUDIO:Hello customer"
    assert meta.media_type == "audio/mpeg"
    assert meta.processing_time_ms >= 0


@pytest.mark.asyncio
async def test_tts_failure_handling():
    pipeline = VoicePipeline(FakeSTT(), FailingTTS())
    with pytest.raises(AgentProcessingError, match="synthesis failed"):
        await pipeline.synthesize("Hello customer")


def test_response_audio_is_keyed_by_message_id():
    store = VoicePlaybackStore()
    store.put("message-1", b"audio-1")
    store.put("message-2", b"audio-2")
    assert store.get("message-1") == b"audio-1"
    assert store.get("message-2") == b"audio-2"
    assert store.get("message-3") is None
