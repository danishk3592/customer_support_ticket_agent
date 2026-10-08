import src

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.config import load_settings
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice.adapters import EdgeTTSService, FasterWhisperSTTService
from src.voice.models import VoiceSynthesisRequest, VoiceTranscriptionResponse
from src.voice.pipeline import VoicePipeline


class Src:
    """Concrete runtime wrapper around the support and voice pipelines.

    This keeps the application startup lifecycle, readiness checks, and request
    orchestration in one place while preserving the existing FastAPI handlers.
    """

    def __init__(self, settings, knowledge_base: Path):
        self.pipeline = SupportPipeline(settings, knowledge_base)
        self.voice_pipeline = VoicePipeline(
            stt_service=FasterWhisperSTTService(settings),
            tts_service=EdgeTTSService(settings),
        )

    @property
    def ready(self) -> bool:
        return self.pipeline.ready

    async def initialize(self) -> None:
        await self.pipeline.initialize()

    async def process(self, session_id: str, message: str) -> ChatResponse:
        if not self.pipeline.ready:
            raise ComponentNotReadyError("Support pipeline is not ready")
        return await self.pipeline.process(session_id, message)

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        return self.pipeline.tickets.get(ticket_id)

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
    ) -> VoiceTranscriptionResponse:
        return await self.voice_pipeline.transcribe(
            audio,
            content_type=content_type,
            filename=filename,
        )

    async def synthesize(self, text: str):
        return await self.voice_pipeline.synthesize(text)


settings = load_settings()
src = Src(settings, Path(__file__).resolve().parents[2] / "knowledge_base")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup owns expensive shared initialization. Request handlers reuse the
    # resulting components, while shutdown makes readiness false immediately.
    await src.initialize()
    yield
    src.pipeline.ready = False


app = FastAPI(title="Customer Support Ticket Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    if not src.ready:
        raise HTTPException(status_code=503, detail="Support pipeline is not ready")
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Keep this transport boundary thin: Pydantic validates the public request,
    # the pipeline owns orchestration, and known service errors are translated
    # to stable HTTP responses here.
    try:
        return await src.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    # Keep this endpoint read-only. It should return the exact repository record,
    # not ask the model to reconstruct ticket details. Test both the successful
    # lookup and unknown-ID response through FastAPI's test client.
    ticket = src.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


_ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
}  # Browser microphone recordings may use WAV or WebM depending on browser.


@app.post("/voice/transcribe", response_model=VoiceTranscriptionResponse)
async def voice_transcribe(file: UploadFile = File(...)) -> VoiceTranscriptionResponse:
    """Convert microphone audio to an editable transcript; it never invokes /chat."""
    if file.content_type and file.content_type.lower() not in _ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type: {file.content_type}. Use a microphone audio format such as WAV or WebM.",
        )
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Audio input is empty")
    try:
        return await src.transcribe(
            audio,
            content_type=file.content_type,
            filename=file.filename,
        )
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/voice/synthesize")
async def voice_synthesize(request: VoiceSynthesisRequest) -> Response:
    """Synthesize the exact displayed assistant text on explicit speaker click."""
    try:
        audio, meta = await src.synthesize(request.text)
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=audio,
        media_type=meta.media_type,
        headers={"X-Processing-Time-Ms": str(meta.processing_time_ms)},
    )
