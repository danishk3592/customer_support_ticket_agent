# Customer Support Ticket Agent

A text-based customer-support agent built from the supplied Zangoh starter repository. It uses FastAPI, Streamlit, LangGraph/LangChain, an OpenAI-compatible endpoint serving an open-source instruction-tuned LLM, ChromaDB, and local Hugging Face embeddings.

## Architecture

```text
Customer
   |
   v
Streamlit UI
   |
   v
FastAPI (/chat, /health, /tickets/{ticket_id})
   |
   v
SupportPipeline
   |
   v
LangGraph
   +-----------------------------+
   |                             |
   v                             v
RAG retrieval                Agent decision
   |                             |
   v                     +-------+-------+
ChromaDB + HF embeddings |               |
   |                     v               v
   |                  answer          ticket
   |                     |               |
   |                     |          SessionStore
   |                     |               |
   |                     |          TicketRepository
   |                     |               |
   +----------> grounded response <-------+
```

### Request flow

1. Streamlit generates one UUID per browser conversation and sends that same `session_id` for every turn.
2. FastAPI validates the public request and delegates to `SupportPipeline`.
3. The LangGraph `retrieve` node searches the local Chroma collection.
4. The `decide` node uses structured LLM output to select exactly `answer` or `ticket` and extracts only values explicitly present in the current turn.
5. The `answer` node uses only retrieved policy context. When retrieval produces no sufficiently relevant evidence, the agent explicitly states that the knowledge base does not contain enough information.
6. The `ticket` node merges validated fields into the isolated session, asks for exactly the next missing field, or creates a ticket through the supplied session-bound tool once `TicketCreate` validates.
7. The repository issues the ticket ID, status, and creation time. Session-level idempotency prevents duplicate tickets.
8. FastAPI returns the response, safe source filenames, and the real ticket ID when one exists.

## Ticket collection guard

The workflow never treats model-inferred ticket fields as authoritative. The model may use conversation history to classify whether a turn belongs to the ticket flow, but ticket fields are accepted only from evidence in the current customer turn or from the exact missing field requested in the immediately continuing workflow. In particular, a word such as `payment` inside an issue description does not automatically satisfy the category field; the customer must explicitly provide a category value or labeled category. Ticket creation is blocked until name, email, issue description, and category all validate.

## Mid-session requirement

The session requirement is implemented through `SessionStore` and the client-supplied `session_id`.

Example:

```text
Turn 1: "My payment was charged twice."
Agent:  "What name should we put on the support ticket?"

Turn 2: "Danish Khan"
Agent:  "What email address should we use for the ticket?"

Turn 3: "danish@example.com"
Agent:  "Please briefly describe the issue..."

Turn 4: "I was charged twice for the same order."
Agent:  "What category fits the issue..."

Turn 5: "payment"
Agent:  "Ticket created successfully. Ticket ID: CST-2026-0001"
```

A Streamlit rerun does not reset the UUID or the displayed message list. The backend stores the ticket fields independently from the browser UI, so a multi-turn ticket can continue after each HTTP request.

If the same completed ticket request is retried with the same `session_id`, `TicketRepository` returns the original ticket instead of creating a second one.

## RAG behavior

The supplied Markdown knowledge base is loaded from `knowledge_base/`, split with `RecursiveCharacterTextSplitter`, embedded using `sentence-transformers/all-MiniLM-L6-v2`, and persisted in Chroma.

The retriever:

- uses deterministic SHA-256 IDs based on source filename and chunk content;
- avoids adding the same chunk again after an API restart;
- uses `settings.rag_top_k` by default;
- applies a relevance threshold of `0.35` using Chroma relevance scores;
- returns only plain `{content, source}` dictionaries;
- sanitizes source values to filenames rather than absolute paths.

The relevance threshold is deliberately conservative for this small knowledge base. It provides a deterministic unknown-answer boundary instead of allowing every nearest-neighbor result to become evidence.

## LLM provider

The starter binds an open-source model through `ChatOpenAI`, configured for any OpenAI-compatible endpoint. The default values are suitable for a local Ollama server:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=not-required
LLM_MODEL=qwen2.5:3b
```

The model can be changed without editing application code by modifying `.env`.

For the decision node, the application first attempts LangChain structured output with a Pydantic schema. When the selected OpenAI-compatible backend does not support structured output, it falls back to a strict JSON response contract and validates the returned object before routing.

## Project structure

```text
customer_support_ticket_agent/
├── knowledge_base/
│   ├── accounts.md
│   ├── payments.md
│   ├── returns.md
│   └── shipping.md
├── docs/
│   ├── RAG_IMPLEMENTATION_GUIDE.md
│   └── DEMO_SCRIPT.md
├── src/
│   ├── api/server.py
│   ├── llm/client.py
│   ├── llm/prompts.py
│   ├── llm/workflow.py
│   ├── rag/document_loader.py
│   ├── rag/embeddings.py
│   ├── rag/retriever.py
│   ├── sessions/store.py
│   ├── tools/ticket_tool.py
│   ├── voice/contracts.py
│   ├── voice/pipeline.py
│   ├── voice/adapters.py
│   └── voice/playback.py
│   ├── config.py
│   ├── models.py
│   └── pipeline.py
├── tests/
├── streamlit_app.py
├── .env.example
└── requirements.txt
```

## Requirements

- Python 3.11 or newer
- An OpenAI-compatible endpoint serving an open-source instruction-tuned model
- Enough local disk space for the embedding model and vector index

## Setup

Create a virtual environment and install the supplied requirements.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Then configure `.env` with the endpoint and model you will use. Do not commit credentials.

## Optional local Ollama setup

If using the default local endpoint, install/run Ollama separately and make sure the configured model is available to that server. Any OpenAI-compatible open-source endpoint may be substituted.

## Run

Start FastAPI from the project root:

```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

Start Streamlit in a second terminal:

```bash
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

- Streamlit: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Port values can be changed through `.env` and corresponding command-line arguments.

## Test

Run the supplied and added tests from the project root:

```bash
pytest -q
```

The test suite covers:

- Markdown loading and source metadata
- deterministic RAG IDs and initialization idempotency
- relevance filtering and unknown-answer boundary
- isolated sessions and conversation history
- ticket validation and repository idempotency
- unique ticket IDs across sessions
- agent routing and multi-turn ticket collection using mocked dependencies
- invalid model decision handling

## Manual API examples

Health:

```bash
curl http://localhost:8000/health
```

Policy question:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-1","message":"How long does standard shipping take?"}'
```

Ticket conversation:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-ticket","message":"My payment was charged twice."}'
```

Continue by sending the requested name, email, issue description, and category using the same `session_id`.

Retrieve a created ticket:

```bash
curl http://localhost:8000/tickets/CST-2026-0001
```

On Windows PowerShell, use `Invoke-RestMethod` or the FastAPI `/docs` interface with the same JSON payloads.


## Voice mid-session integration

The project now adds microphone STT and on-demand TTS without replacing the existing text-agent workflow. The browser microphone records audio, `POST /voice/transcribe` converts it to text using the local faster-whisper adapter, Streamlit shows an editable transcript, and only an explicit **Submit transcript** action sends the confirmed text through the existing `POST /chat` endpoint with the same session ID. Every assistant message receives its own speaker control; clicking it sends the exact displayed response text to `POST /voice/synthesize`, and the returned MP3 is cached by chat-message ID. Voice failures are recoverable and never remove the text response or ticket details.

Voice provider choices and implementation details are documented in `docs/VOICE_IMPLEMENTATION.md`. No high-level conversational platform is used.

### Voice environment variables

```env
STT_MODEL=base
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
TTS_VOICE=en-US-AriaNeural
```

The STT model loads lazily. The first microphone transcription may take longer than later requests.

## Troubleshooting

### `/health` returns 503

The FastAPI lifespan has not completed successfully. Check the API terminal for the startup exception, then verify the Python environment, knowledge-base directory, embedding dependencies, and Chroma persistence path.

### Chat returns 502

The request reached FastAPI, but a downstream agent/model/retrieval/ticket operation failed. Verify the configured LLM endpoint and model, then retry after confirming that the model server is reachable.

### The first startup is slow

The local sentence-transformer embedding model may need to be downloaded once. Subsequent starts reuse the model cache and the persistent Chroma data.

### The answer says the knowledge base does not contain enough information

This is expected for questions whose retrieved evidence does not cross the configured `0.35` relevance threshold. The behavior prevents unsupported model-memory answers.

## Submission checklist

Before packaging the submission:

- include all source code and tests;
- include `README.md`, `.env.example`, and the architecture explanation;
- exclude `.env`, `.venv`, `.data`, caches, generated files, and secrets;
- run `pytest -q` successfully in a clean environment;
- demonstrate at least four representative queries;
- demonstrate a multi-turn ticket creation and ticket retrieval;
- demonstrate the unknown-answer path and duplicate-ticket protection;
- place the final ZIP and the demonstration video in the required Google Drive folder.
