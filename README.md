# RAG-Based-Reseach-Assistant
A Retrieval-Augmented Generation (RAG) system for natural language querying of medical research papers on brain tumor detection. Users can ask questions and receive accurate, citation-grounded answers sourced directly from peer-reviewed PDFs.

![Streamlit](/Images/StreamlitDemo1.png)
![Streamlit](/Images/api-docs.png)


## Overview

This project implements an end-to-end RAG pipeline using a FastAPI backend and a Streamlit frontend. Research papers are parsed from PDF, chunked, embedded using OpenAI models, and stored in a ChromaDB vector database for semantic retrieval. Queries are answered using retrieved context with session-based conversational memory.

**Dataset**
- 5 research papers (91 pages)
- 457 indexed text chunks
- Average response time: 2–5 seconds


## Features

- Natural language question answering over medical PDFs  
- Retrieval-Augmented Generation with source citations  
- Session-based conversational memory  
- Async REST API with multi-user support  
- Interactive web interface  


## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn  
- **AI / RAG:** OpenAI API (GPT-4o-mini, text-embedding-3-small), LangChain, ChromaDB  
- **Frontend:** Streamlit  
- **Data & Tooling:** PyPDF, Git, virtual environments, environment variables


## Project Structure

RAG_Research_Assistant/
├── backend/main.py
├── frontend/streamlit_app.py
├── documents/
├── chroma_db/
└── requirements.txt

---
## Notes:
This project is intended for research and educational use only and is not suitable for clinical or diagnostic decision-making.




