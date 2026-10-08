# Demonstration Script

Target length: 4–5 minutes, which satisfies the five-minute demonstration requirement in the supplied implementation guide while keeping the recording focused.

## 0:00–0:30 — Architecture

Show the README architecture diagram and briefly explain:

```text
Streamlit -> FastAPI -> SupportPipeline -> LangGraph
                                          |-> RAG / Chroma
                                          |-> SessionStore
                                          \-> TicketRepository
```

Mention that the client supplies one `session_id` for the whole conversation.

## 0:30–1:10 — Policy question

Query:

> How long does standard shipping take?

Expected behavior:

- answer is grounded in `shipping.md`;
- source list shows `shipping.md`;
- no ticket is created.

## 1:10–1:40 — Unknown question

Query:

> What is your loyalty-points policy?

Expected behavior:

- no unsupported factual answer is fabricated;
- the UI/API explains that the supplied knowledge base does not contain enough information;
- sources remain empty.

## 1:40–3:20 — Multi-turn ticket creation

Use one conversation/session and enter:

1. `My payment was charged twice.`
2. `Danish Khan`
3. `danish@example.com`
4. `I was charged twice for the same order.`
5. `payment`

Show that the agent asks focused follow-ups for missing fields and only calls the ticket tool after all required values validate.

Expected result:

```text
Ticket created successfully.
Ticket ID: CST-2026-0001
```

Also point out that the same `session_id` is preserved in the background.

## 3:20–3:45 — Mid-session requirement

In the video, explain that Streamlit stores the UUID in `st.session_state`, sends the same UUID on every request, while FastAPI stores the canonical ticket fields in `SessionStore`.

This means the conversation can move across multiple HTTP requests without losing previously collected fields.

## 3:45–4:15 — Duplicate protection

Retry the same completed ticket request in the same session.

Expected result:

- the existing ticket ID is returned;
- a second ticket is not created.

## 4:15–4:45 — Ticket retrieval

Open:

```text
GET /tickets/{ticket_id}
```

Show the JSON response containing the ticket ID, category, summary, customer details, status, and creation timestamp.

## 4:45–5:00 — Tests

Run:

```bash
pytest -q
```

Show the successful test result and finish by mentioning that `.env`, caches, local databases, and secrets are excluded from the submission ZIP.


## Voice mid-session add-on

Include this at the end of the recording:

1. Click the microphone control and record a real support request.
2. Show the generated transcript. Edit one word if appropriate.
3. Click **Submit transcript** and show that the request enters the normal text-agent workflow.
4. On the resulting assistant response, click its `🔊 Speak response` control.
5. Play the generated audio and briefly state that TTS is on-demand, message-specific, cached in-session, and does not replace the text response on failure.
