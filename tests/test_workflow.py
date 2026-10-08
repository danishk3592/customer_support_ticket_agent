from types import SimpleNamespace

import pytest

from src.llm.workflow import DecisionOutput, build_support_workflow
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    async def search(self, query: str, limit=None):
        return list(self.chunks)


class FakeModel:
    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        prompt = str(messages[-1].content)
        if "Return a JSON object" in prompt:
            current = prompt.split("Current customer turn:", 1)[-1].strip().lower()
            if current.startswith("my payment") or current == "payment" or "create a ticket" in current or "ticket" in current:
                return DecisionOutput(
                    route="ticket",
                    issue_description="My payment was charged twice",
                    category="payment",
                )
            return DecisionOutput(route="answer")
        return SimpleNamespace(content="Standard delivery usually takes three to five business days.")


@pytest.mark.asyncio
async def test_workflow_answers_from_retrieved_context() -> None:
    workflow = build_support_workflow(
        FakeModel(),
        retriever=FakeRetriever([{"content": "Standard delivery is 3–5 business days.", "source": "shipping.md"}]),
        sessions=SessionStore(),
        tickets=TicketRepository(),
    )
    result = await workflow.ainvoke(
        {"session_id": "answer-1", "customer_message": "How long does standard shipping take?", "messages": []}
    )
    assert result["route"] == "answer"
    assert result["sources"] == ["shipping.md"]
    assert "three to five" in result["response_text"]


@pytest.mark.asyncio
async def test_workflow_collects_ticket_fields_across_turns() -> None:
    sessions = SessionStore()
    tickets = TicketRepository()
    retriever = FakeRetriever([])
    workflow = build_support_workflow(FakeModel(), retriever=retriever, sessions=sessions, tickets=tickets)

    result = await workflow.ainvoke(
        {
            "session_id": "ticket-1",
            "customer_message": "My payment was charged twice",
            "messages": [],
        }
    )
    assert result["route"] == "ticket"
    assert "name" in result["response_text"].lower()

    session = sessions.get_or_create("ticket-1")
    session.customer_name = "Danish Khan"
    session.customer_email = "danish@example.com"
    session.issue_description = "I was charged twice for the same order."
    session.category = "payment"

    result = await workflow.ainvoke(
        {
            "session_id": "ticket-1",
            "customer_message": "payment",
            "messages": [],
        }
    )
    assert result["ticket_id"].startswith("CST-2026-")
    assert len(list(tickets.all())) == 1

    retry = await workflow.ainvoke(
        {
            "session_id": "ticket-1",
            "customer_message": "create a ticket",
            "messages": [],
        }
    )
    assert retry["ticket_id"] == result["ticket_id"]
    assert len(list(tickets.all())) == 1

@pytest.mark.asyncio
async def test_ticket_does_not_create_until_issue_and_category_are_collected() -> None:
    sessions = SessionStore()
    tickets = TicketRepository()
    workflow = build_support_workflow(
        FakeModel(), retriever=FakeRetriever([]), sessions=sessions, tickets=tickets
    )

    result = await workflow.ainvoke({
        "session_id": "ticket-guard-1",
        "customer_message": "My payment was charged twice.",
        "messages": [],
    })
    assert result["ticket_id"] is None
    assert "name" in result["response_text"].lower()
    assert len(list(tickets.all())) == 0

    result = await workflow.ainvoke({
        "session_id": "ticket-guard-1",
        "customer_message": "Danish Khan",
        "messages": [],
    })
    assert result["ticket_id"] is None
    assert "email" in result["response_text"].lower()
    assert len(list(tickets.all())) == 0

    result = await workflow.ainvoke({
        "session_id": "ticket-guard-1",
        "customer_message": "danish@example.com",
        "messages": [],
    })
    assert result["ticket_id"] is None
    assert "describe" in result["response_text"].lower()
    assert len(list(tickets.all())) == 0

    result = await workflow.ainvoke({
        "session_id": "ticket-guard-1",
        "customer_message": "I was charged twice for the same order.",
        "messages": [],
    })
    assert result["ticket_id"] is None
    assert "category" in result["response_text"].lower()
    assert len(list(tickets.all())) == 0

    result = await workflow.ainvoke({
        "session_id": "ticket-guard-1",
        "customer_message": "payment",
        "messages": [],
    })
    assert result["ticket_id"].startswith("CST-2026-")
    assert len(list(tickets.all())) == 1


class JsonFallbackModel(FakeModel):
    def with_structured_output(self, schema):
        raise RuntimeError("provider does not support structured output")

    async def ainvoke(self, messages):
        prompt = str(messages[-1].content)
        if "Return a JSON object" in prompt:
            return SimpleNamespace(
                content='{"route":"answer","customer_name":null,"customer_email":null,"issue_description":null,"category":null}'
            )
        return SimpleNamespace(content="Grounded answer")


@pytest.mark.asyncio
async def test_workflow_falls_back_to_strict_json_when_needed() -> None:
    workflow = build_support_workflow(
        JsonFallbackModel(),
        retriever=FakeRetriever([{"content": "Policy text", "source": "accounts.md"}]),
        sessions=SessionStore(),
        tickets=TicketRepository(),
    )
    result = await workflow.ainvoke(
        {"session_id": "fallback-1", "customer_message": "What is the policy?", "messages": []}
    )
    assert result["route"] == "answer"
    assert result["response_text"] == "Grounded answer"
