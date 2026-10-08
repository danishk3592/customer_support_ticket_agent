from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import server
from src.models import ChatResponse


class FakePipeline:
    ready = True

    def __init__(self):
        self.tickets = SimpleNamespace(get=lambda ticket_id: None)

    async def initialize(self):
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        return ChatResponse(session_id=session_id, response=f"Echo: {message}")


def test_chat_endpoint_validates_and_delegates(monkeypatch):
    monkeypatch.setattr(server, "pipeline", FakePipeline())
    with TestClient(server.app) as client:
        response = client.post("/chat", json={"session_id": "api-1", "message": "hello"})
        assert response.status_code == 200
        assert response.json()["response"] == "Echo: hello"

        bad = client.post("/chat", json={"session_id": "", "message": "hello"})
        assert bad.status_code == 422


def test_unknown_ticket_is_404(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(server, "pipeline", fake)
    with TestClient(server.app) as client:
        response = client.get("/tickets/does-not-exist")
        assert response.status_code == 404


class FakeVoicePipeline:
    class FakeMeta:
        media_type = "audio/mpeg"
        processing_time_ms = 12

    async def transcribe(self, audio, *, content_type=None, filename=None):
        from src.voice.models import VoiceTranscriptionResponse
        return VoiceTranscriptionResponse(transcript="My payment was charged twice", processing_time_ms=12)

    async def synthesize(self, text):
        return b"fake-audio", self.FakeMeta()


def test_voice_transcription_endpoint(monkeypatch):
    monkeypatch.setattr(server, "pipeline", FakePipeline())
    monkeypatch.setattr(server, "voice_pipeline", FakeVoicePipeline())
    with TestClient(server.app) as client:
        response = client.post(
            "/voice/transcribe",
            files={"file": ("voice.wav", b"fake-wav", "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["transcript"] == "My payment was charged twice"


def test_empty_audio_validation(monkeypatch):
    monkeypatch.setattr(server, "pipeline", FakePipeline())
    monkeypatch.setattr(server, "voice_pipeline", FakeVoicePipeline())
    with TestClient(server.app) as client:
        response = client.post(
            "/voice/transcribe",
            files={"file": ("voice.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()


def test_voice_synthesis_endpoint(monkeypatch):
    monkeypatch.setattr(server, "pipeline", FakePipeline())
    monkeypatch.setattr(server, "voice_pipeline", FakeVoicePipeline())
    with TestClient(server.app) as client:
        response = client.post("/voice/synthesize", json={"text": "Ticket created successfully."})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/mpeg")
        assert response.content == b"fake-audio"


def test_unsupported_voice_audio_type(monkeypatch):
    monkeypatch.setattr(server, "pipeline", FakePipeline())
    monkeypatch.setattr(server, "voice_pipeline", FakeVoicePipeline())
    with TestClient(server.app) as client:
        response = client.post(
            "/voice/transcribe",
            files={"file": ("note.txt", b"not-audio", "text/plain")},
        )
        assert response.status_code == 415
        assert "Unsupported audio type" in response.json()["detail"]
