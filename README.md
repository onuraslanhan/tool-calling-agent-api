# Tool-Calling Agent API

A FastAPI-based tool-calling agent system built on top of a Retrieval-Augmented Generation pipeline.

The system allows a Gemini-powered agent to decide which tool should be used for each question. Instead of always performing document retrieval, the agent can choose to search uploaded documents, perform a calculation, use both tools in sequence, or respond without using a tool when appropriate.

## Live API

The API is deployed and available through Render.

[Open the Tool-Calling Agent API Swagger Documentation](https://tool-calling-agent-api.onrender.com/docs)

The Swagger UI can be used to test all available API endpoints directly from the browser.

## GitHub Repository

[View the project on GitHub](https://github.com/onuraslanhan/tool-calling-agent-api)

## Project Overview

This project combines FastAPI, LangChain, ChromaDB, Google Gemini, Google embeddings, and document loaders to create an agent capable of dynamically selecting tools.

The system supports three types of document sources:

* PDF files
* DOCX files
* Web pages

Uploaded content is processed, split into smaller chunks, converted into embeddings, and stored in a persistent ChromaDB vector database.

The agent has access to two tools:

* `search_documents`
* `calculator`

The important part of the architecture is that the application does not force the same workflow for every question.

For example, a mathematical question does not need to search the uploaded documents. A question about information contained in an uploaded document does not need the calculator. A question can also require both tools.

The agent makes this decision dynamically.

## Architecture

The general architecture of the application is:

```text
User Question

Agent

Available Tools

search_documents
calculator

Agent decides which tool is required

Document Search
Calculation
Multiple Tool Calls
No Tool Call

Final Answer
```

The tool selection and ordering are determined by the LLM.

For example, a document-based question can result in:

```text
Question

search_documents

Retrieved information

Final answer
```

A mathematical question can result in:

```text
Question

calculator

Calculated result

Final answer
```

A more complex question can result in:

```text
Question

search_documents

Retrieved value

calculator

Calculated result

calculator

Final result

Final answer
```

## Why an Agent?

The previous RAG architecture always performed retrieval before generating an answer.

That approach works well for document question answering, but it becomes inefficient when the application needs to handle different types of questions.

For example:

```text
What is 847 divided by 7?
```

There is no reason to search the document database.

The agent can recognize that the question requires a calculation and call only the calculator.

For a document question:

```text
According to the uploaded documents, when was Python first released?
```

The agent can recognize that information must be retrieved from the uploaded documents and call `search_documents`.

For a more complex question:

```text
Find the release year of Python in the uploaded documents,
calculate how many years have passed until 2026,
then convert that number into months.
```

The agent can use multiple tools in sequence.

This behavior is not implemented as a fixed workflow in the application.

The model determines which operations are necessary.

## Tools

### `search_documents`

The `search_documents` tool performs semantic search over the uploaded documents.

It can search content originating from:

* PDF files
* DOCX files
* Web pages

The tool uses ChromaDB similarity search.

The application retrieves the top five results:

```python
retrieved_docs_with_scores = store.similarity_search_with_relevance_scores(
    query,
    k=5
)
```

The returned results are then filtered using a relevance threshold.

The current filtering logic is:

```python
score >= max_score * 0.7 and score > 0.02
```

This means that a retrieved chunk must have a score that is at least 70 percent of the best retrieved score and must also pass the minimum score requirement.

This prevents unrelated document chunks from being unnecessarily passed to the agent.

The tool also includes the source of each retrieved document in its response.

Example:

```text
[Source: https://example.com]

Retrieved document content...
```

### `calculator`

The `calculator` tool evaluates simple mathematical expressions.

Examples include:

```text
15 * 3
```

```text
100 / 4
```

```text
2026 - 1991
```

```text
35 * 12
```

The calculator is exposed as a LangChain tool, allowing the agent to decide when a calculation is required.

The current implementation uses Python's `eval()` with built-ins disabled:

```python
eval(expression, {"__builtins__": {}}, {})
```

This is sufficient for the scope of this personal project, but a production implementation should use a dedicated mathematical expression parser instead.

## Tool Chaining

One of the main features of this project is tool chaining.

The output of one tool can become the input for another tool.

For example, consider this question:

```text
Find the year Python was first released according to the documents,
then calculate how many years have passed since then if the current
year is 2026. Also tell me what that result multiplied by 12 equals.
```

Assume the uploaded document contains:

```text
Python was first released in 1991.
```

The agent can perform the following operations:

```text
search_documents

Result:
1991

calculator

2026 - 1991

Result:
35

calculator

35 * 12

Result:
420
```

The final answer can therefore contain:

```text
Python was first released in 1991.

35 years have passed between 1991 and 2026.

35 years is equivalent to 420 months.
```

The important aspect is that the sequence of operations is not hard-coded.

The application simply provides the tools to the agent.

The LLM decides which tools to call and when.

## Document Ingestion

The API supports PDF, DOCX, and URL ingestion.

### PDF Processing

PDF files are processed using LangChain's `PyPDFLoader`.

```python
loader = PyPDFLoader(file_path)
docs = loader.load()
```

The extracted content is then divided into smaller chunks.

The current configuration is:

```python
chunk_size = 1000
chunk_overlap = 150
```

The resulting chunks are inserted into ChromaDB.

### DOCX Processing

DOCX files are processed using:

```python
Docx2txtLoader(file_path)
```

The extracted content is passed through the same text splitting process.

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
```

The resulting chunks are stored in ChromaDB.

### URL Processing

The URL ingestion endpoint first attempts to extract clean web content using `trafilatura`.

```python
downloaded = trafilatura.fetch_url(url)

if downloaded:
    text = trafilatura.extract(downloaded)
```

If usable content cannot be extracted, the application falls back to LangChain's `WebBaseLoader`.

This allows the API to handle web pages using two different extraction strategies.

## Vector Database

The project uses ChromaDB as its persistent vector database.

The database is stored locally in:

```text
./agent_chroma_db
```

The vector store is initialized through:

```python
get_or_create_vector_store()
```

The function creates the ChromaDB instance only when it is required.

The same vector store is then reused by the API.

## Embeddings

The project uses Google's embedding model through LangChain.

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="embedding-001"
)
```

The embedding model converts document chunks into vectors that can be stored and searched in ChromaDB.

The same embedding configuration is used when performing semantic search.

## Language Model

The agent uses Google's Gemini model through LangChain.

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2,
    max_retries=3
)
```

The relatively low temperature is intended to provide more consistent responses while still allowing the model to dynamically select tools.

## Agent Configuration

The tools are registered with the agent:

```python
tools = [
    search_documents,
    calculator
]
```

The agent is then created with:

```python
agent_executor = create_agent(
    llm,
    tools,
    system_prompt=system_prompt
)
```

The system prompt is especially important because it prevents the model from answering document-related questions using its own general knowledge.

The current instruction is effectively:

```text
Only answer using information returned by the available tools.

Do not use general knowledge.

If the requested information is not available in the uploaded
documents, state that the information is not available.

Use the calculator only for numerical calculations.
```

This keeps the system closer to a document-grounded agent rather than a general-purpose chatbot.

## Important Limitation Discovered During Testing

During testing, the agent initially had access to Gemini's general knowledge.

For example, if the user asked:

```text
How many Ballon d'Or awards does Messi have?
```

the model could answer the question correctly even if no uploaded document contained that information.

This was not the desired behavior.

The goal of the project is for document-based information to come from the uploaded documents rather than from the model's own knowledge.

The system prompt was therefore updated to explicitly prevent this behavior.

After the change, if the requested information cannot be found in the uploaded documents, the agent is instructed to say that the information is not available instead of using its general knowledge.

## API Endpoints

The application currently provides four main endpoints.

### `POST /upload-pdf/`

Uploads and indexes a PDF document.

The file is temporarily saved, processed with `PyPDFLoader`, split into chunks, and inserted into ChromaDB.

Example response:

```json
{
    "message": "'document.pdf' processed and indexed successfully!"
}
```

### `POST /upload-docx/`

Uploads and indexes a DOCX document.

Example response:

```json
{
    "message": "'document.docx' processed and indexed successfully!"
}
```

### `POST /upload-url/`

Downloads and indexes content from a web page.

The URL is provided as form data.

Example:

```text
https://example.com/article
```

Example response:

```json
{
    "message": "'https://example.com/article' processed and indexed successfully!"
}
```

### `POST /ask-agent/`

Sends a question to the agent.

Example question:

```text
According to the uploaded documents, what is Python?
```

The agent decides whether it needs to call `search_documents`, `calculator`, both tools, or neither.

Example response:

```json
{
    "question": "According to the uploaded documents, what is Python?",
    "answer": "..."
}
```

## Swagger API Documentation

The deployed API provides an interactive Swagger UI.

You can open it here:

https://tool-calling-agent-api.onrender.com/docs

The Swagger interface allows the endpoints to be tested directly from the browser without requiring a separate API client.

## Example Workflow

A typical workflow looks like this.

First, upload a PDF:

```text
POST /upload-pdf/
```

The document is processed and indexed in ChromaDB.

Then upload a DOCX if additional information is required:

```text
POST /upload-docx/
```

A web page can also be indexed:

```text
POST /upload-url/
```

After the documents are indexed, send a question:

```text
POST /ask-agent/
```

The agent determines which operation is necessary.

For a document question:

```text
search_documents
```

For a calculation:

```text
calculator
```

For a question requiring both:

```text
search_documents
calculator
```

The final answer is returned through the API.

## Installation

Clone the repository:

```bash
git clone https://github.com/onuraslanhan/tool-calling-agent-api.git
```

Enter the project directory:

```bash
cd tool-calling-agent-api
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project directory.

The application loads the environment file using:

```python
env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)
```

The Google API key must be configured in the environment.

Example:

```env
GOOGLE_API_KEY=your_google_api_key
```

Do not commit the `.env` file to GitHub.

## Running Locally

Start the FastAPI application with Uvicorn:

```bash
uvicorn main:app --reload
```

The API documentation will then be available through:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to upload documents and send questions.

## Tech Stack

The project uses the following technologies:

| Technology                      | Purpose                      |
| ------------------------------- | ---------------------------- |
| FastAPI                         | REST API framework           |
| LangChain                       | Agent and tool framework     |
| Gemini                          | Large language model         |
| Google Generative AI Embeddings | Document embeddings          |
| ChromaDB                        | Persistent vector database   |
| PyPDFLoader                     | PDF processing               |
| Docx2txtLoader                  | DOCX processing              |
| WebBaseLoader                   | Web page loading fallback    |
| Trafilatura                     | Clean web content extraction |
| RecursiveCharacterTextSplitter  | Document chunking            |
| Python                          | Application language         |
| Render                          | API deployment               |

## Project Structure

The project is intentionally kept relatively small because the main goal is to demonstrate tool-calling agent architecture.

A simplified structure looks like:

```text
tool-calling-agent-api/
│
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── agent_chroma_db/
```

The main application contains:

* FastAPI endpoints
* Document ingestion
* Text splitting
* ChromaDB initialization
* Document retrieval
* Calculator tool
* Agent configuration
* Agent question endpoint

## Testing

The project was tested with several different types of questions.

### Test 1

Question:

```text
Find the release year of Python from the documents,
then calculate how many years have passed until 2026,
then multiply that result by 12.
```

Expected behavior:

```text
search_documents
calculator
calculator
```

Result:

```text
1991
35 years
420 months
```

The complete tool chain worked correctly.

### Test 2

Question:

```text
What is 847 divided by 7?
```

Expected behavior:

```text
calculator
```

Result:

```text
121
```

The agent did not need to perform document retrieval.

### Test 3

Question:

```text
What's the weather like today?
```

The current application does not provide a real-time weather tool.

Therefore, the system cannot retrieve live weather information.

This demonstrates an important property of the architecture: the agent can only perform operations for which tools are available.

## Current Limitations

### Tool call information is not returned

The current `/ask-agent/` endpoint returns only:

```json
{
    "question": "...",
    "answer": "..."
}
```

The individual tool calls are currently visible primarily through server-side execution logs.

A future version could return a tool trace such as:

```json
{
    "question": "...",
    "answer": "...",
    "tools_used": [
        "search_documents",
        "calculator",
        "calculator"
    ]
}
```

This would make the agent's reasoning process easier to inspect from the API client.

### Calculator implementation

The calculator currently uses Python's `eval()` with built-ins disabled.

Although this reduces the available Python functionality, it is still not the preferred solution for a production system.

A future version should use a proper expression parser.

### Limited number of tools

The agent currently has only two tools:

```text
search_documents
calculator
```

The architecture can be extended with additional tools such as:

```text
web_search
weather
current_time
database_query
file_search
```

The agent setup itself would not need to be redesigned. New tools can be added to the available tool list.

### Retrieval threshold

The document search uses a relative relevance threshold.

The current logic is:

```text
score >= 70% of the highest retrieved score
```

This works for the current project but can become less predictable as the number and diversity of documents increase.

A future version could use a more sophisticated retrieval and reranking strategy.

## Future Improvements

Possible improvements include:

* Returning complete tool-call traces in the API response.
* Adding a web search tool.
* Adding a real-time date and time tool.
* Adding a weather tool.
* Replacing `eval()` with a dedicated mathematical expression parser.
* Adding authentication.
* Adding rate limiting.
* Adding request logging.
* Adding automated tests.
* Adding document deletion endpoints.
* Adding document metadata management.
* Adding document collection management.
* Adding streaming responses.
* Adding better retrieval and reranking.
* Adding an interface for managing uploaded documents.

## Relation to the Previous RAG Project

This project is a continuation of the Multi-Source RAG API project.

Previous project:

[Multi-Source RAG API](https://github.com/onuraslanhan/rag-pdf-qa-api)

The previous architecture focused primarily on document retrieval and question answering.

This project keeps the document ingestion and vector search concepts but changes the retrieval architecture.

Instead of always performing:

```text
Question

Document Search

LLM

Answer
```

the new system gives the LLM control over the available tools.

This makes the architecture more flexible and allows multiple operations to be combined dynamically.

## Live Demo

The deployed Swagger API can be accessed here:

[Tool-Calling Agent API Swagger UI](https://tool-calling-agent-api.onrender.com/docs)

The source code is available here:

[GitHub Repository](https://github.com/onuraslanhan/tool-calling-agent-api)

## License

This project is intended as a personal and educational project demonstrating FastAPI, RAG, LangChain agents, vector databases, and LLM tool calling.
