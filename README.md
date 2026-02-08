# RAG Document Question Answering System

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents (PDF, TXT, CSV) and ask natural language questions. The system retrieves relevant content from the documents and generates accurate, context-based answers using a Large Language Model.

---

## Features
- Upload and process **PDF, TXT, and CSV** files
- Semantic search using **vector embeddings**
- Context-aware question answering
- Displays source references used for answers
- Interactive **Streamlit** web interface

---

## What is Retrieval-Augmented Generation (RAG)?
RAG is a technique where:
1. Relevant information is **retrieved** from documents using vector search
2. The retrieved content is **added as context**
3. A language model **generates answers** grounded in that context

This reduces hallucinations and enables Q&A over private or custom data.

---

## Tech Stack
- **Python**
- **LangChain**
- **ChromaDB** (Vector Database)
- **Sentence Transformers** (Embeddings)
- **Groq LLM**
- **Streamlit**

---

