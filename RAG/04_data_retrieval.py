"""Step 4: Retrieve the top matching chunks for a query using the FAISS index.

Author: Rakesh Kumar Mali

"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
EMBEDDED_DIR = BASE_DIR / "embedded_data"

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3
INDEX_PATH = EMBEDDED_DIR / "faiss_index.bin"
METADATA_PATH = EMBEDDED_DIR / "metadata.json"

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve the top matching chunks for a query.")
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
    results = retrieve(client, query)

    print(f'\nTop {len(results)} match(es) for: "{query}"\n')
    for result in results:
        print(f"[{result['rank']}] score={result['score']:.4f} | source={result['source']} | chunk_id={result['chunk_id']}")
        print(f"    {result['text']}\n")


if __name__ == "__main__":
    main()
