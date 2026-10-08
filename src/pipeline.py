from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        await self.retriever.initialize()
        self.workflow = build_support_workflow(
            self.model,
            retriever=self.retriever,
            sessions=self.sessions,
            tickets=self.tickets,
        )
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        clean_session_id = session_id.strip()
        clean_message = message.strip()
        if not clean_session_id:
            raise AgentProcessingError("session_id must not be blank")
        if not clean_message:
            raise AgentProcessingError("message must not be blank")

        session = self.sessions.get_or_create(clean_session_id)
        session.history.append({"role": "user", "content": clean_message})

        initial_state = {
            "session_id": clean_session_id,
            "customer_message": clean_message,
            "messages": [
                {"role": item["role"], "content": item["content"]}
                for item in session.history
            ],
            "ticket_id": session.ticket_id,
        }

        try:
            result = await self.workflow.ainvoke(initial_state)
        except (ComponentNotReadyError, AgentProcessingError):
            raise
        except Exception as exc:
            raise AgentProcessingError("The support agent could not process the request") from exc

        response_text = str(result.get("response_text", "")).strip()
        if not response_text:
            raise AgentProcessingError("The support agent returned an empty response")

        sources = result.get("sources", []) or []
        safe_sources = [Path(str(source)).name for source in sources if str(source).strip()]
        ticket_id = result.get("ticket_id")
        if ticket_id and self.tickets.get(str(ticket_id)) is None:
            raise AgentProcessingError("The agent returned an invalid ticket identifier")

        session.history.append({"role": "assistant", "content": response_text})
        return ChatResponse(
            success=True,
            session_id=clean_session_id,
            response=response_text,
            sources=list(dict.fromkeys(safe_sources)),
            ticket_id=str(ticket_id) if ticket_id else None,
        )
