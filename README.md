# rag-from-basic

A Python-based RAG (Retrieval-Augmented Generation) pipeline, built step by step.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r RAG/requirements.txt
```

Copy [RAG/.env.example](RAG/.env.example) to `RAG/.env` and set your `OPENAI_API_KEY` (embedding steps) and `ANTHROPIC_API_KEY` (generation step). `RAG/.env` is gitignored.

## Pipeline steps

### 1. Data extraction — `RAG/01_data_extraction.py`

Extracts text from all PDF files in [RAG/Training_Data](RAG/Training_Data) and writes one `.txt` file per PDF to [RAG/extracted_data](RAG/extracted_data).

```bash
python RAG/01_data_extraction.py
```

- Input: `RAG/Training_Data/*.pdf`
- Output: `RAG/extracted_data/<pdf_name>.txt`
- Uses [pypdf](https://pypi.org/project/pypdf/) for text extraction.

### 2. Data chunking — `RAG/02_data_chunking.py`

Splits each extracted `.txt` file into ~500-character chunks (50-character overlap to preserve context across chunk boundaries) and writes them as JSON to [RAG/chunked_data](RAG/chunked_data).

```bash
python RAG/02_data_chunking.py
```

- Input: `RAG/extracted_data/*.txt`
- Output: `RAG/chunked_data/<file_name>.json` — a list of `{chunk_id, source, chunk_index, text}` records
- Pure standard library, no additional dependencies.

### 3. Data embedding — `RAG/03_data_embedding.py`

Embeds every chunk in [RAG/chunked_data](RAG/chunked_data) using the OpenAI embeddings API and indexes the vectors with [FAISS](https://github.com/facebookresearch/faiss) for similarity search. Output is written to [RAG/embedded_data](RAG/embedded_data).

```bash
python RAG/03_data_embedding.py
```

- Input: `RAG/chunked_data/*.json`
- Output:
  - `RAG/embedded_data/faiss_index.bin` — a FAISS `IndexFlatIP` (cosine similarity via L2-normalized vectors)
  - `RAG/embedded_data/metadata.json` — chunk records in the same order as the vectors in the index, used to map search results back to their source text
- Requires `OPENAI_API_KEY` in `RAG/.env`. Uses the `text-embedding-3-small` model.

> **Note:** Local embedding models (`sentence-transformers`, `fastembed`) were tried first but their native DLLs (`torch`, `onnxruntime`) failed to load on this Windows environment (`WinError 1114`). The pipeline uses the OpenAI API instead.

### 4. Data retrieval — `RAG/04_data_retrieval.py`

Takes a query, embeds it with the same OpenAI model used in step 3, and searches the FAISS index in [RAG/embedded_data](RAG/embedded_data) for the top 3 matching chunks.

```bash
python RAG/04_data_retrieval.py "your question here"
# or run with no argument to be prompted for a query
python RAG/04_data_retrieval.py
```

- Input: `RAG/embedded_data/faiss_index.bin`, `RAG/embedded_data/metadata.json`
- Output: prints the top 3 matches to the console, each numbered `[1]`–`[3]` with its `score` (cosine similarity from the FAISS `IndexFlatIP` index — closer to `1.0` means more similar), source file, chunk ID, and chunk text.
- Requires `OPENAI_API_KEY` in `RAG/.env`.

### 5. Data augmentation — `RAG/05_data_augmentation.py`

Visualizes the augmentation step of RAG: retrieves the top 3 chunks for a query (same logic as step 4) and merges them with the query into the final prompt that would be sent to an LLM. Prints each stage — query, retrieved chunks, augmented prompt — and saves the result to [RAG/augmented_data](RAG/augmented_data).

```bash
python RAG/05_data_augmentation.py "your question here"
# or run with no argument to be prompted for a query
python RAG/05_data_augmentation.py
```

- Input: `RAG/embedded_data/faiss_index.bin`, `RAG/embedded_data/metadata.json`
- Output: `RAG/augmented_data/<timestamp>_<query_slug>.json` — `{query, retrieved_chunks, augmented_prompt}`
- Requires `OPENAI_API_KEY` in `RAG/.env`.

### 6. Data generation — `RAG/06_data_generation.py`

Generates the final answer with Claude (`claude-opus-5` via the [Anthropic API](https://docs.claude.com/)), grounded strictly in the retrieved context. Prompts you to choose an input source, then prints the answer.

```bash
python RAG/06_data_generation.py
```

- Input (interactive prompt — choose one):
  1. **Direct query** — retrieves fresh context via the same logic as step 4 (requires `OPENAI_API_KEY`)
  2. **Augmented file name** — loads `query` + `retrieved_chunks` from an existing file in `RAG/augmented_data` (from step 5), skipping retrieval
- Output: prints the answer to the console.
- The system prompt instructs Claude to answer **only** from the provided context; if the context doesn't contain the answer, it replies with exactly `"No details found related to your question"`.
- Requires `ANTHROPIC_API_KEY` in `RAG/.env`.