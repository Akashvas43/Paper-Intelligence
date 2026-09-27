import sys
import os
import re
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import create_chunks

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone


load_dotenv()


# -----------------------------
# Models / Clients
# -----------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

pinecone_index = pc.Index("research-rag")


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# -----------------------------
# Build retriever for a document
# -----------------------------

def build_retriever(pdf_path, doc_id, records=None):

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

        vectors = embeddings.embed_documents(
            texts
        )

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

def build_multi_retriever(pdf_paths):
    all_chunks = []
    all_vectors = []
    all_texts = []

    for pdf_path in pdf_paths:

        doc_id = os.path.splitext(
            os.path.basename(pdf_path)
        )[0]

        print("Loading:", doc_id)

        pages = load_pdf(pdf_path)

        chunks = create_chunks(
            pages,
            doc_id
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        vectors = embeddings.embed_documents(
            texts
        )

        all_chunks.extend(chunks)
        all_vectors.extend(vectors)
        all_texts.extend(texts)

    tokenized_texts = [
        text.lower().split()
        for text in all_texts
    ]

    bm25 = BM25Okapi(
        tokenized_texts
    )

    return {
        "chunks": all_chunks,
        "vectors": all_vectors,
        "bm25": bm25,
        "doc_id": None
    }


# -----------------------------
# Search
# -----------------------------

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


    # -----------------------------
    # Query Embedding
    # -----------------------------

    start = time.perf_counter()

    query_vector = embeddings.embed_query(
        query
    )

    embedding_time = (
        time.perf_counter() - start
    )


    # -----------------------------
    # Dense Retrieval - Pinecone
    # -----------------------------

    start = time.perf_counter()

    if doc_id is None:

        pinecone_results = pinecone_index.query(
            vector=query_vector,
            top_k=10,
            include_metadata=True
        )

    else:

        if doc_id:
            pinecone_results = pinecone_index.query(
                vector=query_vector,
                top_k=10,
                include_metadata=True,
                filter={"doc_id": {"$eq": doc_id}}
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


    dense_ranking = []

    for match in pinecone_results["matches"]:

        match_doc_id = match["metadata"]["doc_id"]
        chunk_id = match["metadata"]["chunk_id"]

        for i in range(len(chunks)):
            if (
                chunks[i]["chunk_id"] == chunk_id
                and chunks[i]["doc_id"] == match_doc_id
            ):
                dense_ranking.append(i)
                break


    # -----------------------------
    # BM25 Retrieval
    # -----------------------------

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


    # -----------------------------
    # RRF
    # -----------------------------

    start = time.perf_counter()

    rrf_scores = {}

    k = 60


    for rank, index in enumerate(
        dense_ranking
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0)
            + 1 / (k + rank + 1)
        )


    for rank, index in enumerate(
        bm25_ranking
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0)
            + 1 / (k + rank + 1)
        )


    rrf_top_indices = sorted(
        rrf_scores,
        key=lambda i: rrf_scores[i],
        reverse=True
    )[:10]


    rrf_time = (
        time.perf_counter() - start
    )


    # -----------------------------
    # Reranking
    # -----------------------------

    reranker_inputs = [
        (
            query,
            chunks[i]["text"]
        )
        for i in rrf_top_indices
    ]


    start = time.perf_counter()

    reranker_scores = reranker.predict(
        reranker_inputs
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


    # -----------------------------
    # MMR-style selection
    # -----------------------------

    start = time.perf_counter()

    selected_chunks = []

    lambda_value = 0.7

    remaining = [
        index
        for index, score in reranked
    ]


    while (
        len(selected_chunks) < final_k
        and remaining
    ):

        best_index = None

        best_score = -float("inf")


        for index in remaining:

            relevance = 0

            for (
                rerank_index,
                rerank_score
            ) in reranked:

                if rerank_index == index:

                    relevance = rerank_score

                    break


            if len(selected_chunks) == 0:

                mmr_score = relevance

            else:

                max_similarity = 0

                for (
                    selected_index,
                    selected_score
                ) in selected_chunks:

                    similarity = cosine_similarity(
                        [vectors[index]],
                        [vectors[selected_index]]
                    )[0][0]


                    if similarity > max_similarity:

                        max_similarity = similarity


                mmr_score = (
                    lambda_value * relevance
                    - (
                        1 - lambda_value
                    ) * max_similarity
                )


            if mmr_score > best_score:

                best_score = mmr_score

                best_index = index


        for (
            rerank_index,
            rerank_score
        ) in reranked:

            if rerank_index == best_index:

                selected_chunks.append(
                    (
                        rerank_index,
                        rerank_score
                    )
                )

                break


        remaining.remove(
            best_index
        )


    mmr_time = (
        time.perf_counter() - start
    )


    # -----------------------------
    # Total Retrieval Time
    # -----------------------------

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


# -----------------------------
# Validate retrieved context
# -----------------------------

def validate_context(
    top_chunks,
    reranked,
    chunks
):

    if not top_chunks:
        return (
            False,
            "No relevant chunks were retrieved."
        )

    if not reranked:
        return (
            False,
            "No reranked results were found."
        )

    retrieved_pages = set()

    for index, score in top_chunks:
        retrieved_pages.add(
            chunks[index]["page_number"]
        )

    if not retrieved_pages:
        return (
            False,
            "No page information found."
        )

    return (
        True,
        "Retrieved context is sufficient."
    )


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(
    query,
    context,
    history=None
):

    conversation = ""


    if history:

        for item in history:

            conversation += f"""
User: {item["user"]}

Assistant: {item["assistant"]}
"""


    prompt = f"""
You are a research assistant.

Use the conversation history to understand follow-up questions.

Answer the user's question using ONLY the provided research-paper context.

For every important claim in your answer, include a citation
using this format:

[Page X]

Use the page number provided with the relevant chunk.

If the context does not contain enough information, say:

"I don't have enough information in the provided context."

Do not make up facts or citations.

Conversation history:

{conversation}

User question:

{query}

Context:

{context}
"""


    start = time.perf_counter()


    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )


    generation_time = (
        time.perf_counter() - start
    )


    print(
        f"\nGemini generation: {generation_time:.4f}s"
    )


    return interaction.output_text


# -----------------------------
# Validate generated answer
# -----------------------------

def validate_answer(
    answer,
    context
):

    if not answer or not answer.strip():

        return (
            False,
            "Generated answer is empty."
        )


    cited_pages = re.findall(
        r"\[Page (\d+)\]",
        answer
    )


    if not cited_pages:

        return (
            False,
            "No page citations found in the answer."
        )


    context_pages = set(
        re.findall(
            r"Page: (\d+)",
            context
        )
    )


    for page in cited_pages:

        if page not in context_pages:

            return (
                False,
                f"Invalid citation: Page {page}"
            )


    return (
        True,
        "Answer validation passed."
    )