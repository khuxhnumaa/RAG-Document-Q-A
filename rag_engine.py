from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def create_rag_chain(vectordb):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )


    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    def ask(query: str):
        docs = retriever.invoke(query)

        if not docs:
            return {"answer": "I don't know.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Use only the context below to answer.
If unsure, say "I don't know".

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [doc.metadata for doc in docs]
        }

    return ask
