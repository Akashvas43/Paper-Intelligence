import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from dotenv import load_dotenv
from pinecone import Pinecone

from ingestion.embeddings import create_embeddings


load_dotenv()


def upload_to_pinecone(pdf_path):

    api_key = os.getenv(
        "PINECONE_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not found in .env"
        )

    pc = Pinecone(
        api_key=api_key
    )

    index = pc.Index(
        "research-rag"
    )

    doc_id = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    print("Creating embeddings...")

    records = create_embeddings(
        pdf_path,
        doc_id
    )

    vectors = []

    for record in records:

        vectors.append({
            "id": (
                f'{doc_id}-{record["chunk_id"]}'
            ),
            "values": record["vector"],
            "metadata": {
                "doc_id": record["doc_id"],
                "chunk_id": record["chunk_id"],
                "page_number": record["page_number"],
                "text": record["text"]
            }
        })

    print("Uploading vectors to Pinecone...")

    response = index.upsert(
        vectors=vectors,
        batch_size=100
    )

    print("Upload complete.")
    print(
        "Vectors uploaded:",
        response.upserted_count
    )

    return doc_id, records