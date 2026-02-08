from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import pandas as pd

def load_and_split(file_path):
    documents = []

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
        documents = loader.load()

    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

        if df.empty:
            return []

        for _, row in df.iterrows():
            text = " | ".join(
                [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
            )
            if text.strip():
                documents.append(Document(page_content=text))


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)
