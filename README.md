# Tool-Calling Agent API

A FastAPI-based agent system built on top of a RAG pipeline — the LLM decides for itself which tools to call (document search, calculation) and in what order, instead of following a fixed hard-coded sequence. Built with LangChain's agent framework, ChromaDB, HuggingFace embeddings, and Google's Gemini model.

This is a follow-up to my [Multi-Source RAG API](../rag-pdf-qa-api) project — same document ingestion pipeline (PDF/DOCX/URL), but the retrieval step is now exposed as a **tool** the LLM chooses to call, rather than always running.

## What it does

- Ingests PDF, DOCX, and web content into a persistent ChromaDB vector store (identical pipeline to the RAG project)
- Exposes two tools to the LLM: `search_documents` (retrieval with the max-relative relevance threshold from the RAG project) and `calculator` (safe expression evaluation)
- The agent decides, per question, whether it needs to call one tool, both tools in sequence, or no tool at all — the model plans this itself, it isn't hard-coded

## Architecture

```
Question
    → Agent (LLM + tool list)
    → Model decides: does this need a tool?
        → search_documents(query) if the answer requires uploaded document content
        → calculator(expression) if the answer requires a numeric computation
        → both, in sequence, if the question requires chaining (e.g. find a number in a document, then compute with it)
        → neither, if the model can answer directly
    → Final answer
```

## Why an agent instead of a fixed pipeline

In the RAG project, retrieval always ran — every question triggered a search, whether or not the question actually needed document content. That's fine for a single-purpose Q&A tool, but it doesn't scale to mixed workloads: a system that also needs to do calculations, or answer general questions, shouldn't force every query through the same fixed steps.

`create_agent(llm, tools)` gives the model a list of tools with natural-language descriptions (the tool's docstring) and lets it plan its own sequence of calls based on the question — including chaining a tool's output into another tool's input.

## Tool-chaining example (tested)

**Question:** *"Find the year Python was first released according to the documents, then calculate how many years have passed since then if the current year is 2026. Also tell me what 2026 minus that year multiplied by 12 equals."*

**What happened:** the agent called `search_documents` to find the release year (1991, from an uploaded Wikipedia page), then called `calculator` with `2026 - 1991`, then called `calculator` again with the result `* 12` — chaining the output of one tool into the input of the next, without that sequence being hard-coded anywhere in the code.

**Answer:** *"Python was first released in 1991. Years passed (by 2026): 2026 − 1991 = 35 years. Number of months: 35 × 12 = 420 months."* — correct on both retrieval and arithmetic.

## Test results

| # | Question | Expected behavior | Result |
|---|---|---|---|
| 1 | Find the release year from documents, then compute years-passed and months-passed | Chain: search → calculate → calculate | Correct chain, correct answer (1991 → 35 years → 420 months) |
| 2 | What is 847 divided by 7? | Use calculator only, no document search needed | Correct (121), no unnecessary document search |
| 3 | What's the weather like today? | Neither tool applies — answer directly or state the limitation | Answered directly: no real-time weather access, suggested a weather site |

## A real limitation found during testing (and fixed)

Initially, the agent wasn't restricted to document content only. Testing with a question unrelated to any uploaded document ("How many Ballon d'Or does Messi have?") showed the agent answering correctly — but from Gemini's own training data, not from `search_documents`. Unlike the RAG project (which explicitly refuses to answer when the relevance threshold isn't met), the agent had no instruction telling it *not* to fall back on general knowledge.

**Fix:** added an explicit system prompt instructing the agent to answer only from tool results and to state that information isn't available in the documents rather than use its own knowledge, even when it "knows" the answer:

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

Re-tested with the same Messi question after the fix — the agent correctly responded that the information isn't available in the uploaded documents, instead of answering from its own knowledge.

## Known gaps / next steps
- No automated logging/tracing of which tools were called per request in the API response itself — currently only visible via `verbose=True` server-side logs, not returned to the client. A production version would return the tool-call trace alongside the answer.
- `calculator` uses Python's `eval()` with `__builtins__` disabled as a safety measure — fine for a small personal project, but a production tool would use a proper expression-parsing library instead of `eval` in any form.
- Only two tools currently (search + calculator). The same pattern extends to more tools (e.g. a web search tool, a date/time tool) without changing the agent-setup code.
- Inherits the same relevance-threshold approach (and its known limitation — score instability as the document collection grows) from the RAG project's `search_documents` logic.

## Stack

FastAPI · LangChain (`create_agent`) · ChromaDB · HuggingFace Embeddings (`all-MiniLM-L6-v2`) · Google Gemini · `trafilatura` · `python-docx2txt`