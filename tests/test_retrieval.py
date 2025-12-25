from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def test_retrieval_basic():
    """
    Sanity test for similarity-based retrieval.
    Ensures that relevant scientific content is returned
    for a canonical Red Sea query.
    """

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectordb = Chroma(
        persist_directory="chroma_redsea",
        embedding_function=embedding
    )

    query = "Why is the Red Sea so saline compared to other seas?"
    results = vectordb.similarity_search(query, k=5)

    assert len(results) == 5

    
    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print(doc.page_content[:400])

if __name__ == "__main__":
    test_retrieval_basic()