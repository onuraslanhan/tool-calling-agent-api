# Tool-Calling Agent API

A FastAPI-based agent that decides for itself which tool to use for each question — search uploaded documents, run a calculation, chain both together, or answer directly — instead of following a fixed hard-coded pipeline. Built with LangChain, ChromaDB, and Google Gemini. Follow-up to my [Multi-Source RAG API](https://github.com/onuraslanhan/rag-pdf-qa-api).

**Live API:** [tool-calling-agent-api.onrender.com/docs](https://tool-calling-agent-api.onrender.com/docs) — interactive Swagger UI, test every endpoint from the browser.

## What it does

- Ingests PDF, DOCX, and web content into a persistent ChromaDB vector store
- Exposes two tools to the LLM: `search_documents` (retrieval with a relevance threshold) and `calculator` (safe expression evaluation)
- The agent decides, per question, whether it needs one tool, both in sequence, or none — the model plans this itself

## Architecture

```
Question → Agent (LLM + tool list) → model decides:
    search_documents(query)   — if the answer needs document content
    calculator(expression)    — if the answer needs a computation
    both, chained             — e.g. find a number in a document, then compute with it
    neither                   — if no tool applies
→ Final answer
```

## Why an agent instead of a fixed RAG pipeline

In the RAG project, retrieval always ran — every question triggered a search, whether or not it needed document content. That doesn't scale to mixed workloads (calculations, general questions, document lookups all in the same system). `create_agent(llm, tools)` gives the model a list of tools with natural-language descriptions and lets it plan its own sequence of calls, including chaining one tool's output into another tool's input.

## Tool chaining (tested)

**Question:** *"Find the year Python was first released according to the documents, then calculate how many years have passed since then if the current year is 2026. Also tell me what that result multiplied by 12 equals."*

**What happened:** `search_documents` → found **1991** in an uploaded Wikipedia page → `calculator("2026 - 1991")` → **35** → `calculator("35 * 12")` → **420**. Three tool calls, correctly chained, with no hard-coded sequence in the code — the model decided the plan.

## A real limitation found during testing (and fixed)

Initially, the agent could fall back on Gemini's own training data for questions unrelated to any uploaded document (e.g. correctly answering "How many Ballon d'Or does Messi have?" despite no document containing that information). This defeats the purpose of a document-grounded system.

**Fix:** an explicit system prompt restricting the agent to tool results only:

```python
system_prompt = (
    "You are a helpful assistant with access to tools: search_documents and calculator. "
    "Only answer using information returned by these tools — never use your own general knowledge. "
    "If search_documents does not return relevant information for a question, "
    "respond that the information is not available in the uploaded documents, "
    "even if you might know the answer from your own training. "
    "Use calculator only for numeric computations, not for looking up facts."
)
agent_executor = create_agent(llm, tools, system_prompt=system_prompt)
```

Re-tested afterward — the agent correctly refused to answer from its own knowledge and stated the information wasn't in the documents.

## Test results

| # | Question | Expected behavior | Result |
|---|---|---|---|
| 1 | Find Python's release year in the documents, compute years and months since | `search_documents` → `calculator` → `calculator` | Correct chain: 1991 → 35 years → 420 months |
| 2 | What is 847 divided by 7? | `calculator` only, no document search | Correct (121), no unnecessary search |
| 3 | What is 84756392 × 193857? | `calculator` — LLM shouldn't estimate large products itself | Correct exact result (16,430,619,883,944) |
| 4 | What's the weather like today? | No tool applies — answer directly, state the limitation | Answered directly, no tool called |
| 5 | How many Ballon d'Or does Messi have? (before system prompt fix) | Should refuse — no document contains this | Incorrectly answered from general knowledge |
| 6 | Same question (after system prompt fix) | Should refuse | Correctly stated the information isn't in the uploaded documents |

## Deployment

The API is containerized with Docker and deployed on [Render](https://render.com).

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "10000"]
```

Running it locally with Docker:

```bash
docker build -t agent-api .
docker run -p 8000:10000 --env-file .env agent-api
```

The container needs a `GOOGLE_API_KEY` environment variable at runtime — passed via `.env` locally, and via Render's environment variable settings in production. This keeps the deployed environment identical to local development, so "works on my machine" isn't a concern.

## Running locally without Docker

```bash
git clone https://github.com/onuraslanhan/tool-calling-agent-api.git
cd tool-calling-agent-api
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GOOGLE_API_KEY=your_google_api_key
```

```bash
uvicorn agent:app --reload
```
Swagger UI at `http://127.0.0.1:8000/docs`.

markdown
## Session memory (tested)
Each request takes a `session_id`. Two requests with the same `session_id` share conversation history via a LangGraph `MemorySaver` checkpointer:
1. "What's the highest year mentioned in the document?" → answer includes a year
2. "Multiply that by 5" (same `session_id`) → correctly resolves "that" to the year from the previous answer
Different `session_id` → no shared context, as expected.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /upload-pdf/` | Upload and index a PDF |
| `POST /upload-docx/` | Upload and index a DOCX file |
| `POST /upload-url/` | Fetch and index a web page (via `trafilatura`, with `WebBaseLoader` fallback) |
| `POST /ask-agent/` | Ask a question with a `session_id` — the agent decides which tool(s) to use and remembers prior turns in that session |

## Known gaps / next steps

- The API response doesn't return which tools were called — only visible in server logs (`verbose=True`). A future version would include a `tools_used` field in the response.
- Only two tools currently. The pattern extends to more (web search, date/time, etc.) without redesigning the agent setup.
- Inherits the RAG project's relevance-threshold approach and its known instability as the document collection grows in size and topic diversity.
- `gemini-embedding-001` free tier has a 100 requests/minute quota; large document uploads (e.g. long Wikipedia pages) can exceed it. Fixed by batching `add_documents` calls with retry/backoff and increasing chunk size.

## Stack

FastAPI · LangChain (`create_agent`) · LangGraph (checkpointer) . ChromaDB · Google Generative AI Embeddings · Google Gemini · `trafilatura` · `python-docx2txt` · Docker · Render
