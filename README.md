# Research Paper Q&A

A RAG-based question-answering system for research papers.

## Features

- Upload a research paper PDF
- Automatic PDF processing and chunking
- BGE embeddings for semantic search
- Pinecone vector database
- BM25 keyword retrieval
- Reciprocal Rank Fusion (RRF)
- BGE reranking
- MMR-style context selection
- Gemini for answer generation
- Page-based answer citations
- Streamlit chat interface

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- BAAI/bge-small-en-v1.5
- Pinecone
- BM25
- BAAI/bge-reranker-base
- Gemini
- LangChain HuggingFace embeddings

## Project Structure

```
app/          Streamlit application
ingestion/    PDF loading, chunking and indexing
retrieval/    Hybrid retrieval and answer generation
evaluation/   Retrieval and generation evaluation
data/         Benchmark dataset
tests/        Tests
```

## How It Works

1. Upload a research paper PDF.
2. The PDF text is extracted and split into chunks.
3. Chunks are converted into embeddings and stored in Pinecone.
4. User questions are searched using semantic and BM25 retrieval.
5. Results are combined using Reciprocal Rank Fusion.
6. Retrieved chunks are reranked using BGE reranker.
7. Relevant context is passed to Gemini.
8. Gemini generates a grounded answer with page citations.
9. The answer is validated before being displayed.

## Evaluation

The project includes separate evaluation scripts for:

- Retrieval performance
- Generation quality
- Citation validation
- Retrieval latency

## Setup
```
.venv\Scripts\activate
pip install -r requirements.txt
```
## Create a .env file:
```
PINECONE_API_KEY=your_key
GEMINI_API_KEY=your_key
```
## Run
```
streamlit run app/app.py
```