from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")
    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_new_sessions_are_isolated() -> None:
    store = SessionStore()
    first = store.get_or_create("one")
    second = store.get_or_create("two")
    first.customer_name = "Asha"
    assert second.customer_name is None


def test_complete_session_has_no_missing_fields() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Payment was charged twice",
        category="payment",
    )
    assert state.missing_ticket_fields() == []


def test_history_preserves_roles_and_order() -> None:
    state = ConversationState()
    state.history.append({"role": "user", "content": "My payment failed"})
    state.history.append({"role": "assistant", "content": "What is your email?"})
    assert [item["role"] for item in state.history] == ["user", "assistant"]
