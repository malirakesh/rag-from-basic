"""Step 1: Extract text from PDF files in Training_Data and save to extracted_data.

Author: Rakesh Kumar Mali

"""

import logging
from pathlib import Path

from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Training_Data"
OUTPUT_DIR = BASE_DIR / "extracted_data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def main() -> None:
    if not INPUT_DIR.exists():
        logger.error("Input folder not found: %s", INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", INPUT_DIR)
        return

    for pdf_path in pdf_files:
        logger.info("Extracting: %s", pdf_path.name)
        text = extract_text_from_pdf(pdf_path)

        output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")
        logger.info("Saved extracted text to: %s", output_path)

    logger.info("Extraction complete. %d file(s) processed.", len(pdf_files))


if __name__ == "__main__":
    main()
