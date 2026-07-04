from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def build_chroma(chunks, persist_dir="chroma_redsea"):
    """Build (and persist) a Chroma vector store from document chunks.

    ChromaDB >= 0.4 persists automatically on write, so an explicit
    .persist() call is no longer required.
    """
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir,
    )
    return vectordb
