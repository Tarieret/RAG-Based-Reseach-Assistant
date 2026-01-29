"""
Script to rebuild the vector database from PDF documents
This ensures the database is created locally and not synced to iCloud
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import shutil

print("=" * 70)
print("RAG RESEARCH ASSISTANT - DATABASE REBUILD")
print("=" * 70)

# Step 1: Remove old database
if os.path.exists("./chroma_db"):
    print("\n[1/4] Removing old database...")
    shutil.rmtree("./chroma_db")
    print("Old database removed")

# Step 2: Load PDFs
print("\n[2/4] Loading PDF documents...")
loader = DirectoryLoader(
    "documents/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)
documents = loader.load()
print(f"Loaded {len(documents)} pages from PDF files")

# Step 3: Chunk documents
print("\n[3/4] Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 4: Create embeddings and vector store
print("\n[4/4] Creating vector database...")
print("This will take 1-2 minutes...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Vector database created with {len(chunks)} chunks!")
print(f"Database saved to './chroma_db'")

# Step 5: Test the database
print("\n" + "=" * 70)
print("TESTING DATABASE")
print("=" * 70)

test_query = "What deep learning architectures are used?"
results = vectorstore.similarity_search(test_query, k=3)

print(f"\nTest Query: '{test_query}'")
print(f"Found {len(results)} relevant chunks:\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"  Page: {doc.metadata.get('page', 'N/A')}")
    print(f"  Preview: {doc.page_content[:150]}...")
    print()

print("=" * 70)
print("DATABASE REBUILD COMPLETE!")
print("=" * 70)
print("\nYou can now start the backend server:")
print("  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000")
