from Ingest.load_docs import load_documents

def test_loading():

    docs = load_documents("Knowledge_Base")

    assert len(docs) > 0, "No documents were loaded"

    sample = docs[0]

    # Text checks
    assert hasattr(sample, "page_content"), "Document has no page_content"
    assert len(sample.page_content.strip()) > 0, "Empty page content"

    # Metadata checks
    assert "source" in sample.metadata, "Missing source metadata"
    assert "page" in sample.metadata, "Missing page metadata"

    print(f"Loaded {len(docs)} pages successfully.")
    print("Sample source:", sample.metadata["source"])
    print("Sample page:", sample.metadata["page"])
    print(sample.page_content[:300])


if __name__ == "__main__":
    test_loading()
