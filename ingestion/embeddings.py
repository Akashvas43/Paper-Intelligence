import os
import sys
import streamlit as st

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from ingestion.chunker import create_chunks
from ingestion.pdf_loader import (
    load_pdf,
    looks_like_research_paper
)

from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )

def create_embeddings(pdf_path, doc_id):

    pages = load_pdf(pdf_path)

    if not looks_like_research_paper(pages):
        raise ValueError(
            "This PDF does not appear to be a research paper."
        )

    chunks = create_chunks(
        pages,
        doc_id
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embedding_model = get_embeddings()

    vectors = []

    batch_size = 32

    for i in range(0, len(texts), batch_size):

        batch = texts[
            i:i + batch_size
        ]

        batch_vectors = (
            embedding_model.embed_documents(
                batch
            )
        )

        vectors.extend(
            batch_vectors
        )

    records = []

    for i in range(len(chunks)):

        records.append({
            "chunk_id": chunks[i]["chunk_id"],
            "doc_id": chunks[i]["doc_id"],
            "page_number": chunks[i]["page_number"],
            "text": chunks[i]["text"],
            "vector": vectors[i]
        })

    return records

if __name__ == "__main__":

    pdf_path = input(
        "Enter PDF path: "
    ).strip()

    doc_id = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    records = create_embeddings(
        pdf_path,
        doc_id
    )

    print("Document:", doc_id)
    print("Records:", len(records))
    print(
        "Vector dimensions:",
        len(records[0]["vector"])
    )