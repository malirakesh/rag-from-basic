"""Step 2: Split extracted text into ~500-character chunks with overlap and save as JSON."""

import json
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "extracted_data"
OUTPUT_DIR = BASE_DIR / "chunked_data"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_length:
            break
        start = end - overlap

    return chunks


def main() -> None:
    if not INPUT_DIR.exists():
        logger.error("Input folder not found: %s", INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        logger.warning("No extracted text files found in %s", INPUT_DIR)
        return

    for txt_path in txt_files:
        logger.info("Chunking: %s", txt_path.name)
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        chunk_records = [
            {
                "chunk_id": f"{txt_path.stem}_{i}",
                "source": txt_path.name,
                "chunk_index": i,
                "text": chunk,
            }
            for i, chunk in enumerate(chunks)
        ]

        output_path = OUTPUT_DIR / f"{txt_path.stem}.json"
        output_path.write_text(json.dumps(chunk_records, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d chunk(s) to: %s", len(chunk_records), output_path)

    logger.info("Chunking complete. %d file(s) processed.", len(txt_files))


if __name__ == "__main__":
    main()
