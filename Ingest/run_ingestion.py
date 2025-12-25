from .load_docs import load_documents
from .clean_docs import filter_pages
from .chunk_docs import chunk_documents
from .build_vectorstore import build_chroma


docs = load_documents("Knowledge_Base")
docs = filter_pages(docs)
chunks = chunk_documents(docs)
vectordb = build_chroma(chunks)

print(f"Loaded {len(docs)} pages")
print(f"Created {len(chunks)} chunks")
