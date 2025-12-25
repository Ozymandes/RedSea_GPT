# Phase II – Data Engineering & Vector Database Construction

## Overview

Phase II transforms the raw documents collected in Phase I into structured, searchable vector representations suitable for semantic retrieval. This phase implements the core data engineering pipeline required for Retrieval-Augmented Generation (RAG), including text processing, chunking, embedding, and vector database persistence.

The emphasis of Phase II is on **semantic coherence, traceability, and retrieval quality**, rather than purely maximizing embedding volume.

---

## Text Processing Pipeline

### Document Loading

PDF documents are loaded on a page-by-page basis to preserve fine-grained provenance information, including source document and page number. Pages with insufficient textual content (e.g., figures or blank pages) are filtered out.

### Chunking Strategy

Initial chunking experiments revealed that overly small or rigid chunks fragmented scientific explanations across multiple vectors, degrading retrieval coherence.

To address this, a **recursive character-based chunking strategy** was adopted with the following configuration:

- Chunk size: ~1,200 characters  
- Chunk overlap: ~150 characters  

This configuration balances semantic completeness with retrieval granularity, ensuring that explanatory units (e.g., physical drivers of salinity or circulation patterns) are preserved within individual chunks.

---

## Embedding and Vector Storage

Each text chunk is embedded using the `sentence-transformers/all-mpnet-base-v2` model, selected for its strong performance on semantic similarity tasks involving scientific and technical language.

Embeddings are stored in a **persistent Chroma vector database**, allowing efficient similarity search and reproducible experimentation across sessions.

The resulting vector database is stored in:


---

## Metadata Schema

Each embedded chunk is associated with structured metadata to support traceability, filtering, and citation:

- **source**: originating document (PDF title or filename)
- **page**: page number within the source document
- **doc_type**: document category (e.g., textbook, review article)
- **domain**: scientific domain (geology, oceanography, ecology)

This schema enables transparent attribution of retrieved content and supports future metadata-based filtering.

---

## Retrieval Validation

Retrieval quality is validated using similarity-based search against canonical domain questions (e.g., *“Why is the Red Sea so saline compared to other seas?”*). A dedicated retrieval test script (`test_retrieval.py`) confirms that top-k results are semantically relevant, originate from authoritative sources, and preserve correct metadata.

Iterative refinement of chunk size and overlap significantly improved retrieval coherence and reduced repetitive or overly narrow results.

---

## Data Quality Summary

- Source documents: 13 PDFs  
- Total pages processed: 1,324  
- Chunking method: Recursive character-based  
- Chunk size: ~1,200 characters  
- Chunk overlap: ~150 characters  
- Total chunks generated: 5,448  
- Embedding model: sentence-transformers/all-mpnet-base-v2  
- Vector database: Chroma (persistent)  
- Deduplication required: No (distinct sources)

---

## Phase II Definition of Done

- ✔ Text cleaned and normalized  
- ✔ Documents chunked with overlap  
- ✔ Embeddings generated successfully  
- ✔ Vector database populated and persisted  
- ✔ Relevant content retrievable via similarity search  
- ✔ Metadata preserved and accessible  

**Phase II is complete.**
