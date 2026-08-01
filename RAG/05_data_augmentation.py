"""Step 5: Visualize the augmentation step - combining retrieved context with the user query.

Author: Rakesh Kumar Mali

"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
EMBEDDED_DIR = BASE_DIR / "embedded_data"
OUTPUT_DIR = BASE_DIR / "augmented_data"

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3
INDEX_PATH = EMBEDDED_DIR / "faiss_index.bin"
METADATA_PATH = EMBEDDED_DIR / "metadata.json"

PROMPT_TEMPLATE = """Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

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


def build_augmented_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{chunk['rank']}] (source: {chunk['source']}, score: {chunk['score']:.4f})\n{chunk['text']}"
        for chunk in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, query=query)


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] or "query"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the augmentation step of a RAG pipeline.")
    parser.add_argument("query", nargs="*", help="Query text. If omitted, you will be prompted.")
    args = parser.parse_args()

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        logger.error("Embedded data not found in %s. Run 03_data_embedding.py first.", EMBEDDED_DIR)
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Add it to RAG/.env")
        return

    query = " ".join(args.query) if args.query else input("Enter your query: ").strip()
    if not query:
        logger.error("No query provided.")
        return

    client = OpenAI(api_key=api_key)
    chunks = retrieve(client, query)
    augmented_prompt = build_augmented_prompt(query, chunks)

    print("\n--- 1. Original query ---")
    print(query)

    print(f"\n--- 2. Retrieved context ({len(chunks)} chunk(s)) ---")
    for chunk in chunks:
        print(f"[{chunk['rank']}] score={chunk['score']:.4f} | source={chunk['source']} | chunk_id={chunk['chunk_id']}")

    print("\n--- 3. Augmented prompt (query + context merged, ready for the LLM) ---")
    print(augmented_prompt)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}_{slugify(query)}.json"
    output_path.write_text(
        json.dumps(
            {"query": query, "retrieved_chunks": chunks, "augmented_prompt": augmented_prompt},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Saved augmentation result to: %s", output_path)


if __name__ == "__main__":
    main()
