# Tool-Calling Agent API

A FastAPI-based agent system built on top of a RAG pipeline. The LLM decides for itself which tools to call, such as document search or calculation, and in what order, instead of following a fixed hard-coded sequence. The project is built using LangChain's agent framework, ChromaDB, Google Gemini Embeddings, and Google's Gemini model.

This is a follow-up to the Multi-Source RAG API project, sharing the same document ingestion pipeline for PDF, DOCX, and URL formats. However, the retrieval step is now exposed as a tool that the LLM chooses to call dynamically rather than running on every query.

## What it does

- Ingests PDF, DOCX, and web content into a persistent ChromaDB vector store.
- Exposes two tools to the LLM: `search_documents`, which performs retrieval using a relative relevance threshold, and `calculator`, which provides safe expression evaluation.
- The agent evaluates each question individually to determine whether it requires one tool, multiple tools in sequence, or no tool at all.

## Architecture

Question
→ Agent (LLM + tool list)
→ Model decides: does this need a tool?
→ search_documents(query) if the answer requires uploaded document content
→ calculator(expression) if the answer requires a numeric computation
→ both, in sequence, if the question requires chaining (e.g. find a number in a document, then compute with it)
→ neither, if the model can answer directly
→ Final answer


## Why an agent instead of a fixed pipeline

In the standard RAG project, retrieval always executed for every query regardless of whether document content was needed. While suitable for dedicated Q&A tools, this approach does not scale well to mixed workloads that require calculations or general reasoning.

Using `create_agent(llm, tools)` supplies the model with a tool list and natural-language docstring descriptions, allowing it to plan execution sequences, including chaining a tool output directly into another tool input.

## Tool-chaining example (tested)

**Question:** *"Find the year Python was first released according to the documents, then calculate how many years have passed since then if the current year is 2026. Also tell me what 2026 minus that year multiplied by 12 equals."*

**Execution flow:** The agent invoked `search_documents` to locate the release year of 1991 from an uploaded source, called `calculator` with `2026 - 1991`, and subsequently invoked `calculator` with the intermediate result multiplied by 12, chaining tool outputs without hard-coded routing.

**Answer:** *"Python was first released in 1991. Years passed (by 2026): 2026 − 1991 = 35 years. Number of months: 35 × 12 = 420 months."*

## Test results

| # | Question | Expected behavior | Result |
|---|---|---|---|
| 1 | Find the release year from documents, then compute years-passed and months-passed | Chain: search → calculate → calculate | Correct chain and calculated values (1991 → 35 years → 420 months) |
| 2 | What is 847 divided by 7? | Use calculator only, no document search needed | Correct output (121) without document retrieval |
| 3 | What's the weather like today? | Neither tool applies — answer directly or state limitations | Handled correctly by stating real-time weather data is unavailable |

## Implementation adjustment for grounding

Initial testing revealed that questions outside the uploaded document scope could trigger fallback to Gemini training data rather than respecting retrieval boundaries. To enforce strict grounding, an explicit system prompt was introduced:

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
Following this update, queries outside the document corpus correctly trigger responses stating the information is absent from the uploaded source material.

Known limitations and next steps
API responses do not currently return structured tool-call traces, which remain visible solely via server-side logs.

The calculator relies on Python eval() with restricted builtins as a simplified mechanism rather than a dedicated expression parser.

The tool suite currently consists of search and calculator utilities, though additional capabilities can be registered without modifying core agent logic.

Inherits score stability constraints associated with threshold-based document retrieval.

Tech stack
FastAPI · LangChain (create_agent) · ChromaDB · Google Gemini Embeddings · Google Gemini · trafilatura · python-docx2txt
