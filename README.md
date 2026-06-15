# Advanced Multi-Source RAG System

This project demonstrates an Advanced Retrieval-Augmented Generation (RAG) pipeline capable of ingesting and retrieving information from multiple heterogeneous data sources.

## Data Sources

* PDF Documents
* DOCX Documents
* CSV Files
* Excel Files
* Website Content
* HTML files

## Architecture

Documents → Chunking → Embeddings → Chroma Vector Database → Semantic Search → Context Retrieval

## Technologies Used

* Python
* LangChain
* HuggingFace Embeddings
* ChromaDB
* Pandas
* BeautifulSoup
* PDF Loaders
* DOCX Loaders

## Features

* Multi-source document ingestion
* Metadata preservation
* Semantic search
* Vector similarity retrieval
* Advanced chunking strategy
* Retrieval across 220+ chunks

## Example Query

Question:
How many maternity leaves an employee entitled to?

Retrieved Result:
Eligible employees may receive up to 26 weeks of paid maternity leave in accordance with company policy.

This demonstrates successful semantic retrieval across multiple document sources.

## Retrieval Example

Query:
How many maternity leaves an employee entitled to?

Retrieved Context:
Eligible employees may receive up to 26 weeks of paid maternity leave in accordance with company policy.
