import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from fastapi import FastAPI, UploadFile, File, Form
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import trafilatura
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="Agent API")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2,
    max_retries=3
)

vector_store = None

def get_or_create_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = Chroma(embedding_function=embeddings, persist_directory="./agent_chroma_db")
    return vector_store

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        store = get_or_create_vector_store()
        add_documents_with_rate_limit(store, splits)
        return {"message": f"'{file.filename}' processed and indexed successfully!"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload-docx/")
async def upload_docx(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        store = get_or_create_vector_store()
        add_documents_with_rate_limit(store, splits)
        return {"message": f"'{file.filename}' processed and indexed successfully!"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def load_url_clean(url: str) -> list[Document]:
    text = None
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
    if not text or len(text.strip()) == 0:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if docs and len(docs[0].page_content.strip()) > 0:
            text = docs[0].page_content
        else:
            raise ValueError(f"Could not get content from URL: {url}")
    return [Document(page_content=text, metadata={"source": url})]

@app.post("/upload-url/")
async def upload_url(url: str = Form(...)):
    docs = load_url_clean(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    store = get_or_create_vector_store()
    add_documents_with_rate_limit(store, splits)
    return {"message": f"'{url}' processed and indexed successfully!"}

@tool
def search_documents(query: str) -> str:
    """Search the uploaded documents (PDF, DOCX, web pages) for information relevant to the query. Use this whenever the question requires facts from the uploaded documents."""
    store = get_or_create_vector_store()
    retrieved_docs_with_scores = store.similarity_search_with_relevance_scores(query, k=5)
    
    if not retrieved_docs_with_scores:
        return "No documents have been uploaded yet."
    
    scores = [score for doc, score in retrieved_docs_with_scores]
    max_score = max(scores)
    
    relevant_docs = [
        doc for doc, score in retrieved_docs_with_scores 
        if score >= max_score * 0.7 and score > 0.02
    ]
    
    if not relevant_docs:
        return "No relevant information found in the uploaded documents."
    
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" 
        for doc in relevant_docs
    )
    return context

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression, e.g. '15 * 3' or '100 / 4'. Use this whenever the question requires a numeric calculation."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def add_documents_with_rate_limit(store, splits, batch_size=20, delay=15):
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        for attempt in range(3):
            try:
                store.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    time.sleep(delay)
                else:
                    raise
        if i + batch_size < len(splits):
            time.sleep(2)

tools = [search_documents, calculator]
system_prompt = (
    "You are a helpful assistant with access to tools: search_documents and calculator. "
    "Only answer using information returned by these tools — never use your own general knowledge. "
    "If search_documents does not return relevant information for a question, "
    "respond that the information is not available in the uploaded documents, "
    "even if you might know the answer from your own training. "
    "Use calculator only for numeric computations, not for looking up facts."
)

checkpointer = MemorySaver()
agent_executor = create_agent(llm, tools, system_prompt=system_prompt, checkpointer=checkpointer)

@app.post("/ask-agent/")
async def ask_agent(question: str = Form(...), session_id: str = Form(default="default")):
    config = {"configurable": {"thread_id": session_id}}
    result = agent_executor.invoke({"messages": [("user", question)]}, config=config)
    final_message = result["messages"][-1]
    
    content = final_message.content
    if isinstance(content, list):
        answer_text = "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        answer_text = content
    
    return {
        "question": question,
        "answer": answer_text
    }   