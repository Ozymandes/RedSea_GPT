# Phase I – Data Acquisition

## Overview

Phase I of the RedSea GPT project focuses on the acquisition of high-quality, authoritative data describing the Red Sea as a natural system. The goal of this phase is to establish a reliable and reproducible corpus that can support downstream preprocessing, embedding, and retrieval without introducing factual noise or ambiguity.

Rather than prioritizing quantity of data, Phase I emphasizes **data quality, provenance, and scientific reliability**, in alignment with the project’s objective of building a grounded Retrieval-Augmented Generation (RAG) system.

---

## Data Sources

The corpus consists of **thirteen authoritative academic publications** focused on the Red Sea’s geology, oceanography, and ecology. These sources include peer-reviewed review volumes and textbooks that are widely cited within Red Sea research.

All sources were collected in their original **PDF format** and preserved without modification to ensure reproducibility and traceability.

### Rationale for Using PDFs

PDF-based academic sources were intentionally selected over web-scraped content for the following reasons:

- **Scientific reliability**: The selected documents are peer-reviewed or editorially curated, minimizing the risk of factual errors.
- **Depth and coherence**: Academic volumes provide comprehensive, system-level explanations that are difficult to replicate using fragmented web pages.
- **Stable provenance**: Each extracted text segment can be traced back to a specific document and page number.
- **Reduced noise**: Excluding web scraping avoids boilerplate text, duplicated content, and inconsistencies common in online sources.

Although web scraping was initially considered, it was ultimately deemed unnecessary for Phase I, as the curated corpus already provides sufficient coverage for the project’s scope.

---

## Data Organization

Raw data is stored in the following directory:


This directory contains all source PDFs in their original form. No preprocessing or modification is applied at this stage.

---

## Phase I Output

By the end of Phase I:

- Authoritative data sources are clearly defined and collected
- Raw documents are preserved intact
- The corpus is ready for preprocessing and vectorization

---

## Phase I Definition of Done

- ✔ Data sources defined and justified  
- ✔ Documents collected and stored locally  
- ✔ Raw data preserved without modification  
- ✔ Data ready for Phase II preprocessing  

**Phase I is complete.**
