from Ingest.load_docs import load_documents
from Ingest.clean_docs import filter_pages
from Ingest.chunk_docs import chunk_documents


def test_chunking():
    """
    Sanity test for document chunking.
    Confirms that chunking produces
    semantically meaningful chunks with metadata.
    """

    docs = load_documents("Knowledge_Base")
    docs = filter_pages(docs)

    chunks = chunk_documents(docs)

    assert len(chunks) > 0, "No chunks were generated"

    sample = chunks[10]

    # Content checks
    assert len(sample.page_content.strip()) > 200, "Chunk content too short"

    # Metadata checks
    assert "source" in sample.metadata, "Missing source metadata in chunk"
    assert "page" in sample.metadata, "Missing page metadata in chunk"

    print(f"Generated {len(chunks)} chunks.")
    print("Sample chunk source:", sample.metadata["source"])
    print("Sample chunk page:", sample.metadata["page"])
    print("Chunk length:", len(sample.page_content))
    print(sample.page_content[:400])


if __name__ == "__main__":
    test_chunking()
