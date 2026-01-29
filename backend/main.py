"""
FastAPI Backend for RAG Research Assistant
Provides REST API endpoints for document querying and retrieval
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from contextlib import asynccontextmanager

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Global variables for RAG components
vectorstore = None
llm = None
retriever = None
prompt = None

# Conversation history storage (in-memory, can be replaced with Redis/DB)
conversation_sessions = {}


class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    question: str
    session_id: Optional[str] = "default"
    k: Optional[int] = 4  # Number of chunks to retrieve


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str
    sources: List[dict]
    session_id: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vectorstore_ready: bool
    llm_ready: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup"""
    global vectorstore, llm, retriever, prompt

    print("🚀 Initializing RAG system...")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
        print("   Set it before making requests")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load vector store
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    print("✅ Vector store loaded")

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    print("✅ LLM initialized")

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create prompt template
    template = """You are a helpful research assistant specializing in medical AI and brain tumor detection.
Use the following pieces of context to answer the question at the end.

If you don't know the answer based on the context provided, just say "I don't have enough information in the provided documents to answer that question." Don't make up an answer.

If the answer is in the context, provide a detailed answer and mention which study or paper it comes from if possible.

Context:
{context}

Question: {question}

Detailed Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    print("✅ RAG system ready!")

    yield

    # Cleanup
    print("🛑 Shutting down RAG system...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Research Assistant API",
    description="API for querying medical research papers using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_docs(docs):
    """Combine retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectorstore_ready": vectorstore is not None,
        "llm_ready": llm is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question

    Parameters:
    - question: The user's question
    - session_id: Optional session ID for conversation history
    - k: Number of document chunks to retrieve (default: 4)

    Returns:
    - answer: Generated answer from the LLM
    - sources: List of source documents used
    - session_id: Session ID for this conversation
    """
    if not vectorstore or not llm:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Update retriever with custom k value
        custom_retriever = vectorstore.as_retriever(search_kwargs={"k": request.k})

        # Retrieve relevant documents
        source_docs = custom_retriever.invoke(request.question)

        # Format context
        context = format_docs(source_docs)

        # Get conversation history for this session
        if request.session_id not in conversation_sessions:
            conversation_sessions[request.session_id] = []

        history = conversation_sessions[request.session_id]

        # Build conversation context from history (last 3 exchanges)
        history_context = ""
        if history:
            recent_history = history[-3:]
            history_context = "\n\nPrevious conversation:\n"
            for i, exchange in enumerate(recent_history, 1):
                history_context += f"Q{i}: {exchange['question']}\n"
                history_context += f"A{i}: {exchange['answer']}\n\n"

        # Combine context with history
        full_context = context + history_context

        # Generate answer
        messages = prompt.format_messages(context=full_context, question=request.question)
        answer = llm.invoke(messages)

        # Store in conversation history
        conversation_sessions[request.session_id].append({
            'question': request.question,
            'answer': answer.content
        })

        # Format sources for response
        sources = []
        for doc in source_docs:
            sources.append({
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "content": doc.page_content[:300]  # First 300 characters
            })

        return {
            "answer": answer.content,
            "sources": sources,
            "session_id": request.session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversation_sessions:
        return {"session_id": session_id, "history": []}

    return {
        "session_id": session_id,
        "history": conversation_sessions[session_id]
    }


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": f"Session {session_id} not found"}


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    total_docs = 0
    if vectorstore:
        try:
            # Get collection stats
            collection = vectorstore._collection
            total_docs = collection.count()
        except Exception as e:
            # If count fails, return a placeholder
            print(f"Warning: Could not count documents: {e}")
            total_docs = "N/A"

    return {
        "total_documents": total_docs,
        "active_sessions": len(conversation_sessions),
        "total_conversations": sum(len(hist) for hist in conversation_sessions.values())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
