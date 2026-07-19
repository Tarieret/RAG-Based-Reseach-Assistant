# RAG-Based Research Assistant

*Researchers lose hours manually searching PDFs for relevant findings; a grounded natural-language interface turns that into a direct question-and-answer, without the hallucination risk that makes most LLM search tools unusable for medical literature.*

**Problem:** Peer-reviewed medical research on brain tumor detection is dense and scattered across dozens of PDFs. Manually searching for specific findings is slow, and a naive LLM wrapper risks hallucinating citations or conclusions, which is unacceptable in a research-integrity context. The goal was natural-language querying that stays strictly grounded in the source papers.

**Architecture decision:** Constrained LLM responses to retrieved context only, treating hallucination as a prompt-engineering problem to eliminate upfront rather than a downstream filtering step. Indexed content with semantic embeddings in ChromaDB instead of keyword search, since dense medical terminology needs conceptual matching, not exact-string matching. Built citation-backed responses as a core requirement rather than a nice-to-have, given the stakes of the domain. Split the system into a FastAPI backend and Streamlit frontend to support async, multi-user access rather than a single-session notebook demo.

**What broke:** 

**Metric:** 20 research papers (478 pages) indexed into 2,184 text chunks, with average query response time of 2–5 seconds. *[subject to change post-rerun]*

## Overview
An end-to-end RAG pipeline enabling natural language querying of peer-reviewed medical research papers on brain tumor detection. Users ask questions and receive accurate, citation-grounded answers sourced directly from indexed PDFs, delivered through a FastAPI backend and Streamlit frontend. The idea came out of a Spring 2025 master's-level project on deep learning for brain tumor detection from MRI images, which shifted the focus from individual models toward supporting the research workflow itself.

![Streamlit](/Images/rag-demo-full.gif)

## Dataset & Scale

| Metric | Value |
|---|---|
| Research papers indexed | 20 papers (478 pages) |
| Text chunks indexed | 2,184 |
| Average response time | 2–5 seconds |

## Features
- Natural language question answering over medical PDFs
- Retrieval-Augmented Generation with source citations
- Session-based conversational memory
- Async REST API with multi-user support
- Interactive Streamlit web interface

## Architecture
PDF Papers → PyPDF Parsing → Text Chunking → OpenAI Embeddings
                                              ↓
User Query → Streamlit → FastAPI → ChromaDB Retrieval → GPT-4o-mini → Response + Citations

## Tech Stack
- **Backend:** Python, FastAPI, Uvicorn
- **AI / RAG:** OpenAI API (GPT-4o-mini, text-embedding-3-small), LangChain, ChromaDB
- **Frontend:** Streamlit
- **Data & Tooling:** PyPDF, Git, virtual environments, environment variables

## Note
This project is intended for research and educational use only and is not suitable for clinical or diagnostic decision-making.

## Reflection
Grounding and citation aren't polish, they're the difference between a demo and a tool someone can actually trust with research decisions.
