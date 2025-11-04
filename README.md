# RAG System - w301b Assignment

Complete PDF RAG (Retrieval-Augmented Generation) system with support for text, images, and tables.

## Features

### Core Functionality
- Elasticsearch Integration: Local deployment with vector search
- PDF Processing: Text extraction, image extraction with AI descriptions, table extraction
- Vector Embeddings: Local Sentence Transformers model
- Vector Search: Semantic similarity-based retrieval
- Answer Generation: Context-aware responses with citations

## Tech Stack

- Python 3.12
- Elasticsearch 8.11 (Docker)
- OpenAI GPT-4o-mini (text generation and image description)
- Sentence Transformers (local embedding model)
- PyMuPDF & pdfplumber (PDF processing)

## Quick Start

### 1. Install Dependencies

    pip install -r requirements.txt

### 2. Start Elasticsearch

    docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.11.0

### 3. Configure Environment

Create .env file and add:

    OPENAI_API_KEY=your_api_key_here

### 4. Run the System

Place your PDF in test_pdf/ directory, then run:

    python pdf_rag.py

## Project Structure

    RAG_w301b/
    ├── pdf_rag.py              # Complete PDF RAG system
    ├── mini_rag_demo.py        # Simplified demo
    ├── simple_test.py          # Basic tests
    ├── config.py               # Configuration
    ├── index_manager.py        # Elasticsearch index management
    ├── requirements.txt        # Dependencies
    └── test_pdf/              # PDF files directory

## System Architecture

1. Document Processing: PDF to Text/Images/Tables extraction and Chunking
2. Embedding Generation: Text chunks to Vector embeddings
3. Indexing: Vectors plus metadata to Elasticsearch
4. Retrieval: Query to Vector search to Top-K documents
5. Generation: Retrieved docs plus Query to LLM to Answer with citations

## Assignment Requirements Completed

- Local Elasticsearch deployment
- PDF text extraction and chunking
- PDF image extraction with AI descriptions
- PDF table extraction and processing
- Vector embedding generation and indexing
- Vector search implementation
- Response generation with citations

## Demo

Run simplified demo (no PDF needed):

    python mini_rag_demo.py

Run full system with PDF:

    python pdf_rag.py

## Key Features

**Text Processing**
- Intelligent chunking with overlap
- Token-based splitting
- Metadata preservation

**Image Processing**
- GPT-4 Vision for image description
- Context augmentation
- Automatic extraction from PDF

**Table Processing**
- Table extraction using pdfplumber
- Markdown conversion
- Content indexing

**Search and Retrieval**
- Vector similarity search
- Semantic understanding
- Top-K retrieval with scoring

## Performance

- Processing: 2-5 minutes for 10-page PDF
- Query Response: 5-10 seconds per query
- Index Size: 10-50MB per 10-page document

## Requirements

- Python 3.8 or higher
- Docker for Elasticsearch
- OpenAI API key
- 8GB RAM or more recommended

## License

Course assignment project for w301b.

## Author

w301b Course Assignment - 2025
