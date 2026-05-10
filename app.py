import streamlit as st
from src.rag_pipeline import RAGPipeline
import tempfile
import os

st.set_page_config(
    page_title="PDF Chat with RAG",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Chat with RAG")
st.markdown("Upload a PDF and ask questions about it using AI.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload your PDF")
    st.markdown("2. PDF is split into chunks")
    st.markdown("3. Chunks are embedded with OpenAI")
    st.markdown("4. Your question retrieves top chunks")
    st.markdown("5. GPT-4 answers using those chunks")
    st.markdown("---")
    st.markdown("Built by [Pramod Gangula](https://github.com/pramod019g-oss)")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF... This may take a moment."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            st.session_state.rag = RAGPipeline(openai_api_key=openai_api_key)
            num_chunks = st.session_state.rag.load_pdf(tmp_path)
            st.session_state.pdf_processed = True
            st.session_state.messages = []
            st.success(f"✅ PDF processed! Created {num_chunks} chunks. Start asking questions below.")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            os.unlink(tmp_path)

if uploaded_file and st.session_state.pdf_processed:
    st.markdown("---")
    st.subheader("💬 Chat")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Source chunks used"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:** {src[:400]}...")

    # Chat input
    if prompt := st.chat_input("Ask anything about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.rag.query(prompt)
                    st.markdown(answer)
                    with st.expander("📎 Source chunks used"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Chunk {i}:** {src[:400]}...")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
