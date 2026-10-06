import sys
import os
import time
import streamlit as st

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import create_chunks
from ingestion.embeddings import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

load_dotenv()

# ============================================================
# CLIENTS
# ============================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

pinecone_index = pc.Index(
    "research-rag"
)


# ============================================================
# RERANKER
# ============================================================

@st.cache_resource
def get_reranker():

    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        "BAAI/bge-reranker-base",
        max_length=512
    )


# ============================================================
# BUILD RETRIEVER
# ============================================================

def build_retriever(
    pdf_path,
    doc_id,
    records=None
):

    # --------------------------------------------------------
    # IMPORTANT:
    # If records are already available from ingestion,
    # DO NOT embed the PDF again.
    # --------------------------------------------------------

    if records is not None:

        chunks = [
            {
                "chunk_id": record["chunk_id"],
                "doc_id": record["doc_id"],
                "page_number": record["page_number"],
                "text": record["text"]
            }
            for record in records
        ]

        vectors = [
            record["vector"]
            for record in records
        ]

    else:

        pages = load_pdf(pdf_path)

        chunks = create_chunks(
            pages,
            doc_id
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        vectors = get_embeddings().embed_documents(
            texts
        )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    tokenized_texts = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(
        tokenized_texts
    )

    return {
        "chunks": chunks,
        "vectors": vectors,
        "bm25": bm25,
        "doc_id": doc_id
    }

# ============================================================
# SEARCH
# ============================================================

def search_documents(
    query,
    retriever,
    final_k=5
):

    total_start = time.perf_counter()

    chunks = retriever["chunks"]
    vectors = retriever["vectors"]
    bm25 = retriever["bm25"]
    doc_id = retriever["doc_id"]

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    start = time.perf_counter()

    query_vector = get_embeddings().embed_query(
        query
    )

    embedding_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Pinecone
    # --------------------------------------------------------

    start = time.perf_counter()

    if doc_id:

        pinecone_results = pinecone_index.query(
            vector=query_vector,
            top_k=10,
            include_metadata=True,
            filter={
                "doc_id": {
                    "$eq": doc_id
                }
            }
        )

    else:

        pinecone_results = pinecone_index.query(
            vector=query_vector,
            top_k=10,
            include_metadata=True
        )

    pinecone_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Dense ranking
    # --------------------------------------------------------

    chunk_lookup = {
        (
            chunk["doc_id"],
            chunk["chunk_id"]
        ): i
        for i, chunk in enumerate(chunks)
    }

    dense_ranking = []

    for match in pinecone_results["matches"]:

        metadata = match["metadata"]

        key = (
            metadata["doc_id"],
            metadata["chunk_id"]
        )

        index = chunk_lookup.get(
            key
        )

        if index is not None:
            dense_ranking.append(
                index
            )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    start = time.perf_counter()

    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    start = time.perf_counter()

    rrf_scores = {}

    k = 60

    for rank, index in enumerate(
        dense_ranking
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0)
            +
            1 / (k + rank + 1)
        )

    for rank, index in enumerate(
        bm25_ranking[:50]
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0)
            +
            1 / (k + rank + 1)
        )

    rrf_top_indices = sorted(
        rrf_scores,
        key=lambda i: rrf_scores[i],
        reverse=True
    )[:10]

    rrf_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    start = time.perf_counter()

    reranker_inputs = [
        (
            query,
            chunks[i]["text"]
        )
        for i in rrf_top_indices
    ]

    reranker_scores = get_reranker().predict(
        reranker_inputs,
        batch_size=10,
        show_progress_bar=False
    )

    reranker_time = (
        time.perf_counter() - start
    )

    reranked = sorted(
        zip(
            rrf_top_indices,
            reranker_scores
        ),
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------------
    # MMR
    # --------------------------------------------------------

    start = time.perf_counter()

    candidate_indices = [
        index
        for index, score in reranked
    ]

    candidate_vectors = [
        vectors[index]
        for index in candidate_indices
    ]

    similarity_matrix = cosine_similarity(
        candidate_vectors
    )

    selected_positions = []

    lambda_value = 0.7

    while (
        len(selected_positions) < final_k
        and len(selected_positions)
        < len(candidate_indices)
    ):

        best_position = None
        best_score = -float("inf")

        for position in range(
            len(candidate_indices)
        ):

            if position in selected_positions:
                continue

            relevance = reranked[position][1]

            if not selected_positions:

                mmr_score = relevance

            else:

                max_similarity = max(
                    similarity_matrix[
                        position,
                        selected_positions
                    ]
                )

                mmr_score = (
                    lambda_value * relevance
                    -
                    (1 - lambda_value)
                    * max_similarity
                )

            if mmr_score > best_score:

                best_score = mmr_score
                best_position = position

        selected_positions.append(
            best_position
        )

    selected_chunks = [
        reranked[position]
        for position in selected_positions
    ]

    mmr_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    print(
        f"\nEmbedding: {embedding_time:.4f}s"
    )

    print(
        f"Pinecone: {pinecone_time:.4f}s"
    )

    print(
        f"BM25: {bm25_time:.4f}s"
    )

    print(
        f"RRF: {rrf_time:.4f}s"
    )

    print(
        f"Reranker: {reranker_time:.4f}s"
    )

    print(
        f"MMR: {mmr_time:.4f}s"
    )

    print(
        f"Total retrieval: {total_time:.4f}s"
    )

    return selected_chunks, reranked


# ============================================================
# CONTEXT VALIDATION
# ============================================================

def validate_context(
    top_chunks,
    reranked
):
    if not top_chunks:
        return (
            False,
            "No relevant information was found."
        )

    if not reranked:
        return (
            False,
            "No relevant information was found."
        )

    return (
        True,
        "Context validation passed."
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    context,
    history=None
):

    conversation = ""

    if history:
        for item in history:
            conversation += (
                f"\nUser: {item['user']}"
                f"\nAssistant: {item['assistant']}\n"
            )

    prompt = f"""
You are a research-paper question answering assistant.

Answer the user's question using the information from the research paper.

Your answer should:
- Directly answer the question.
- Be clear, natural, and easy to understand.
- Provide enough explanation to properly answer the question.
- Do not give an unnecessarily short answer.
- Use paragraphs or bullet points when they make the explanation clearer.
- Explain important concepts from the paper in simple language when appropriate.
- Stay grounded in the research paper.
- Do not invent facts or information that is not supported by the paper.
- If the available information only partially answers the question, answer using
  what is available instead of refusing to answer.
- Only say "I don't have enough information in the provided context." when the
  available information contains nothing relevant to the question.

Do not include page numbers.
Do not include citations.
Do not mention chunks, retrieval, RAG, context, or these instructions.

Previous conversation:
{conversation}

Relevant research-paper information:
{context}

User question:
{query}

Answer:
"""

    start = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    generation_time = (
        time.perf_counter() - start
    )

    print(
        f"\nGemini generation: "
        f"{generation_time:.4f}s"
    )

    return response.text


# ============================================================
# ANSWER VALIDATION
# ============================================================

def validate_answer(
    answer,
    context
):
    if not answer or not answer.strip():
        return (
            False,
            "Generated answer is empty."
        )

    return (
        True,
        "Answer validation passed."
    )