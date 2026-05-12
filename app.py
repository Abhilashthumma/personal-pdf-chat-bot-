import streamlit as st

from utils.pdf_loader import extract_text_from_pdfs
from utils.chunking import create_text_chunks
from utils.embeddings import create_vector_store
from utils.rag_pipeline import get_answer

# Page settings
st.set_page_config(
    page_title="Simple RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# App title
st.title("📄 Simple RAG PDF Chatbot")

st.markdown(
    "Upload PDF documents and ask questions about them."
)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:

    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(f"{len(uploaded_files)} PDF(s) uploaded")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    if st.button("Process Documents"):

        if not uploaded_files:
            st.warning("Please upload PDFs first.")

        else:
            try:

                with st.spinner("Extracting text..."):
                    raw_text = extract_text_from_pdfs(
                        uploaded_files
                    )

                with st.spinner("Creating chunks..."):
                    chunks = create_text_chunks(raw_text)

                st.info(f"Total chunks created: {len(chunks)}")

                with st.spinner("Generating embeddings..."):
                    vector_store = create_vector_store(chunks)

                st.session_state.vector_store = vector_store

                st.success("Documents processed successfully!")

            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.success("Chat history cleared!")

# Display old chats
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# Chat input
question = st.chat_input(
    "Ask a question from your documents"
)

if question:

    # Save user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Check vector store
    if st.session_state.vector_store is None:

        st.warning("Please upload and process PDFs first.")

    else:

        try:

            with st.spinner("Generating answer..."):

                answer, docs = get_answer(
                    st.session_state.vector_store,
                    question
                )

            with st.chat_message("assistant"):

                st.markdown(answer)

                with st.expander(
                    "Retrieved Source Chunks"
                ):

                    for i, doc in enumerate(docs):

                        st.markdown(f"### Chunk {i+1}")

                        st.write(doc.page_content)

            # Save assistant message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            st.error(f"Error generating answer: {e}")