import streamlit as st
import tempfile

from document_processor import load_and_split
from vector_store import create_vector_store
from rag_engine import create_rag_chain

st.set_page_config(page_title="RAG Document Q&A")

st.title("RAG Document Question Answering")

uploaded_file = st.file_uploader(
    "Upload a PDF, TXT, or CSV file",
    type=["pdf", "txt", "csv"]
)

if uploaded_file:
    import os

    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.info("Processing document...")
    chunks = load_and_split(file_path)

    if not chunks:
        st.error("No readable content found in the uploaded file")
        st.stop()

    vectordb = create_vector_store(chunks)
    qa_chain = create_rag_chain(vectordb)
    st.success("Document processed successfully!")

    query = st.text_input("Ask a question from the document")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain(query)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Source References")
        for src in result["sources"]:
            st.write(src)
