import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from ingestion.chunker import create_chunks
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_loader import (
    load_pdf,
    looks_like_research_paper
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

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = embeddings.embed_documents(texts)

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

    pdf_path = input("Enter PDF path: ").strip()

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