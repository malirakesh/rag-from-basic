"""Step 3: Embed chunked data using the OpenAI embeddings API and index with FAISS."""

import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "chunked_data"
OUTPUT_DIR = BASE_DIR / "embedded_data"

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
INDEX_PATH = OUTPUT_DIR / "faiss_index.bin"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_chunks() -> list[dict]:
    chunks = []
    for json_path in sorted(INPUT_DIR.glob("*.json")):
        chunks.extend(json.loads(json_path.read_text(encoding="utf-8")))
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.info("Requesting embeddings for chunks %d-%d...", i, i + len(batch) - 1)
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def main() -> None:
    if not INPUT_DIR.exists():
        logger.error("Input folder not found: %s", INPUT_DIR)
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        logger.error("OPENAI_API_KEY not set. Add it to RAG/.env")
        return

    chunks = load_chunks()
    if not chunks:
        logger.warning("No chunked data found in %s", INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)
    texts = [chunk["text"] for chunk in chunks]
    logger.info("Embedding %d chunk(s) with model: %s", len(texts), EMBEDDING_MODEL)
    embeddings = np.array(embed_texts(client, texts), dtype="float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved FAISS index (%d vectors, dim=%d) to: %s", index.ntotal, dimension, INDEX_PATH)
    logger.info("Saved chunk metadata to: %s", METADATA_PATH)


if __name__ == "__main__":
    main()
