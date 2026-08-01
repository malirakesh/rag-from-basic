"""Step 6: Generate a grounded answer from retrieved/augmented context using Claude."""

import json
import logging
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
EMBEDDED_DIR = BASE_DIR / "embedded_data"
AUGMENTED_DIR = BASE_DIR / "augmented_data"

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "claude-opus-5"
MAX_TOKENS = 1024
TOP_K = 3
INDEX_PATH = EMBEDDED_DIR / "faiss_index.bin"
METADATA_PATH = EMBEDDED_DIR / "metadata.json"

NOT_FOUND_MESSAGE = "No details found related to your question"
SYSTEM_PROMPT = f"""You are a retrieval-augmented assistant. Answer the user's question using ONLY the information in the context below.
Do not use any outside knowledge, and do not guess.
If the context does not contain enough information to answer the question, respond with exactly this sentence and nothing else: "{NOT_FOUND_MESSAGE}"."""

load_dotenv(BASE_DIR / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    embedding = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(embedding)
    return embedding


def retrieve(client: OpenAI, query: str, top_k: int = TOP_K) -> list[dict]:
    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    query_embedding = embed_query(client, query)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx == -1:
            continue
        chunk = metadata[idx]
        results.append({"rank": rank, "score": float(score), **chunk})
    return results


def load_from_file(filename: str) -> tuple[str, list[dict]]:
    data = json.loads((AUGMENTED_DIR / filename).read_text(encoding="utf-8"))
    return data["query"], data["retrieved_chunks"]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{chunk['rank']}] (source: {chunk['source']}, score: {chunk['score']:.4f})\n{chunk['text']}"
        for chunk in chunks
    )


def generate_answer(client: Anthropic, query: str, context: str) -> str:
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )

    if response.stop_reason == "refusal":
        return NOT_FOUND_MESSAGE

    for block in response.content:
        if block.type == "text":
            return block.text
    return NOT_FOUND_MESSAGE


def main() -> None:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY not set. Add it to RAG/.env")
        return

    print("How would you like to provide input?")
    print("  1. Enter a query directly")
    print("  2. Provide an augmented file name (from RAG/augmented_data)")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "2":
        filename = input("Augmented file name: ").strip()
        file_path = AUGMENTED_DIR / filename
        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            return
        query, chunks = load_from_file(filename)
    else:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            logger.error("Embedded data not found in %s. Run 03_data_embedding.py first.", EMBEDDED_DIR)
            return

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not set. Add it to RAG/.env")
            return

        query = input("Enter your query: ").strip()
        if not query:
            logger.error("No query provided.")
            return

        openai_client = OpenAI(api_key=openai_api_key)
        chunks = retrieve(openai_client, query)

    context = build_context(chunks)
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    answer = generate_answer(anthropic_client, query, context)

    print(f'\nQuery: "{query}"\n')
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
