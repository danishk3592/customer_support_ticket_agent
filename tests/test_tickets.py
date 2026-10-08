from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository


def request() -> TicketCreate:
    return TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    repository = TicketRepository()
    first = repository.create("session-1", request())
    second = repository.create("session-1", request())
    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_repository_creates_unique_ids_across_sessions() -> None:
    repository = TicketRepository()
    first = repository.create("session-1", request())
    second = repository.create("session-2", request())
    assert first.ticket_id != second.ticket_id


def test_repository_preserves_ticket_fields() -> None:
    repository = TicketRepository()
    ticket = repository.create("session-1", request())
    assert ticket.category == "payment"
    assert ticket.customer_email == "test@example.com"
    assert ticket.status == "open"
    assert ticket.created_at is not None
