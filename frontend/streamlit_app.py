"""
Streamlit Frontend for RAG Research Assistant
Beautiful UI for querying medical research papers
"""

import streamlit as st
import requests
import json
from typing import List, Dict
import time

# API Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2ecc71;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def query_api(question: str, session_id: str, k: int = 4) -> Dict:
    """Send query to FastAPI backend"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "question": question,
                "session_id": session_id,
                "k": k
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def get_session_history(session_id: str) -> List[Dict]:
    """Get conversation history from API"""
    try:
        response = requests.get(f"{API_URL}/sessions/{session_id}/history")
        response.raise_for_status()
        return response.json().get("history", [])
    except:
        return []


def clear_session(session_id: str):
    """Clear session history"""
    try:
        requests.delete(f"{API_URL}/sessions/{session_id}")
    except:
        pass


def get_stats() -> Dict:
    """Get system statistics"""
    try:
        response = requests.get(f"{API_URL}/stats")
        response.raise_for_status()
        return response.json()
    except:
        return {}


def display_source(source: Dict, index: int, conversation_idx: int = 0):
    """Display a source document with formatting"""
    filename = source['source'].split('/')[-1]
    page = source['page']
    content = source['content']

    with st.expander(f"Source {index}: {filename} (Page {page})"):
        st.text_area(
            "Content Preview",
            content,
            height=150,
            key=f"source_{conversation_idx}_{index}",
            disabled=True
        )


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">RAG Research Assistant</div>', unsafe_allow_html=True)
    st.markdown("### Explore AI-driven brain tumor detection research")

    # Check API health
    if not check_api_health():
        st.error("Cannot connect to API backend. Make sure FastAPI is running on http://localhost:8000")
        st.info("Run: `cd backend && python main.py` or `uvicorn backend.main:app --reload`")
        st.stop()

    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Session management
        st.subheader("Session Management")
        st.text_input("Session ID", value=st.session_state.session_id, disabled=True)

        if st.button("New Session"):
            st.session_state.session_id = f"session_{int(time.time())}"
            st.session_state.chat_history = []
            clear_session(st.session_state.session_id)
            st.rerun()

        if st.button("Clear History"):
            st.session_state.chat_history = []
            clear_session(st.session_state.session_id)
            st.success("History cleared!")

        # Retrieval settings
        st.subheader("Retrieval Settings")
        k_value = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=4,
            help="How many document chunks to use for answering"
        )

        # System stats
        st.subheader("System Stats")
        stats = get_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('total_documents', 'N/A'))
            with col2:
                st.metric("Sessions", stats.get('active_sessions', 'N/A'))
            st.metric("Total Queries", stats.get('total_conversations', 'N/A'))

        # Example questions
        st.subheader("Example Questions")
        example_questions = [
            "What deep learning architectures are used?",
            "What are the challenges in brain tumor diagnosis?",
            "What datasets were used in the studies?",
            "What is the accuracy of the models?",
            "How does MRI help in brain tumor detection?"
        ]

        for eq in example_questions:
            if st.button(eq, key=f"example_{eq}"):
                st.session_state.example_question = eq

    # Main content area
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for i, exchange in enumerate(st.session_state.chat_history):
            # Question
            with st.chat_message("user"):
                st.write(exchange['question'])

            # Answer
            with st.chat_message("assistant"):
                st.markdown(exchange['answer'])

                # Sources
                if 'sources' in exchange and exchange['sources']:
                    with st.expander(f"View {len(exchange['sources'])} Sources"):
                        for idx, source in enumerate(exchange['sources'], 1):
                            display_source(source, idx, i)

    # Query input
    st.subheader("Ask a Question")

    # Check if example question was clicked
    default_question = ""
    if 'example_question' in st.session_state:
        default_question = st.session_state.example_question
        del st.session_state.example_question

    question = st.text_area(
        "Enter your question",
        value=default_question,
        height=100,
        placeholder="E.g., What deep learning architectures are used for brain tumor detection?"
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        submit_button = st.button("Search", type="primary")

    with col2:
        if st.button("Copy Last Answer"):
            if st.session_state.chat_history:
                last_answer = st.session_state.chat_history[-1]['answer']
                st.code(last_answer, language=None)

    # Process query
    if submit_button and question.strip():
        with st.spinner("Searching documents and generating answer..."):
            # Query the API
            result = query_api(question, st.session_state.session_id, k_value)

            if result:
                # Store in chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result['sources']
                })

                # Rerun to display new message
                st.rerun()

    elif submit_button:
        st.warning("Please enter a question!")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>Built with Streamlit and FastAPI</p>
            <p>Powered by OpenAI GPT-4 and LangChain</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
