*This project has been created as part of the 42 curriculum by lmuler-f.*

# RAG Against the Machine

## Description
This project implements a Retrieval-Augmented Generation (RAG) pipeline from scratch. The goal is to ingest a massive codebase (vLLM), index it, retrieve the most relevant snippets for user queries, and feed them into a local language model (`Qwen3-0.6B`) to generate accurate, context-grounded answers.

## Instructions
### Installation
This project strictly uses `uv` as the package manager.
```bash
# Install dependencies
make install
```
### Execution

The system provides a CLI built with `fire`. Below are all the required commands:

**1. Index the corpus**
Ingests files from `data/raw/` and builds the index.
```bash
make run ARGS="index --max_chunk_size 2000"
```

**2. Search a single query**
Returns the top-K sources for a single question.
```bash
make run ARGS="search 'What is vLLM?' --k 5"
```

**3. Search a full dataset**
Runs batch search over a dataset and outputs a JSON file with results.
```bash
make run ARGS="search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions"
```

**4. Generate an answer for a single query**
Retrieves context and uses the LLM to generate an answer.
```bash
make run ARGS="answer 'What hardware platforms does vLLM support?' --k 5"
```

**5. Generate answers for a full dataset**
Reads search results, generates answers, and outputs the final JSON.
```bash
make run ARGS="answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions"
```

**6. Internal Evaluation**
Reports our internal implementation of Recall@K against a ground-truth dataset (independent of the official moulinette).
```bash
make run ARGS="evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json"
```

## System Architecture

The architecture is divided into three decoupled layers:

1. **Ingestion & Indexing:** Reads raw files and builds a BM25 inverted index.

2. **Retrieval:** A fast search engine that tokenizes queries and fetches the top-K relevant `DocumentChunks`.

3. **Generation:** Utilizes Hugging Face's `transformers` to load `Qwen3-0.6B` and generate answers based strictly on the retrieved context using a specialized Chat Template.
Data contracts between layers are strictly validated using `pydantic`.


## Chunking Strategy

To maximize retrieval accuracy, we use `langchain-text-splitters`.

- **Python Code:** Parsed using AST-aware tules(`RecursiveCharacterTextSplitter.from_language`), ensuring classes and funcions are not arbitrarily broken.
- **Markdown/Text:** Split by headers and paragraphs. Both strategies a chunk_overlap of 200 characters to prevent cutting context at critical boundaries, while strictly tracking `first_character_index` and `last_character_index`.

## Retrieval Method
I implemented **Lexical Searching using BM25** (via `rank-bm25`).
Unlike standard TF-IDF, BM25 natively handles term frequency saturation and document length normalization. I applied a custom regex tokenizer `[a-z0-9_]+` to preserve crucial programming identifiers (e.g., oreserving `lm_eval` instead of splitting it into `lm` and `eval`), which drastically improved code-related query matching.

## Performance Analysis
Evaluated using the official `moulinette` against the public datasets (`max_chunk_size=2000`, `k=10`):
- **Indexing Time:** ~6 seconds for 1,969 files / 14,512 chunks (well within the 300s limit).
- **Retrieval Throughput:** ~8–9s per 100 questions cold (~17s for 200 questions, well within the 90s limit) and < 0.01s warm (via query caching).
- **Docs Recall@5:** **84.0%** (Recall@1: 64.0%, Recall@3: 78.0%, Recall@10: 90.0%) — exceeds the 80% threshold.
- **Code Recall@5:** **79.8%** (Recall@1: 47.5%, Recall@3: 71.7%, Recall@10: 87.9%) — exceeds the 50% threshold.
- **Tokenizer Impact Experiment:** Switching from simple punctuation splitting to a code-aware regex tokenizer (preserving full identifiers, splitting CamelCase/snake_case sub-tokens, and indexing relative file paths) boosted Code Recall@5 from **50.5% to 79.8%**.

## Bonus Features
1. **Caching (`src/cache.py`):**
   - **In-memory index caching:** `BM25Retriever` caches the loaded `index.pkl` in RAM (`_INDEX_CACHE`), eliminating repeated disk deserialization across calls (`~0.0001s` warm load).
   - **Persistent query caching:** Search results and LLM-generated answers are hashed (`SHA-256` of namespace, `k`, and query) and cached in `data/processed/query_cache.json`. Repeated `/answer` API calls drop from several seconds on CPU to `< 0.05s`.
2. **Local HTTP API (`src/api.py`):**
   - Built with `FastAPI` and `uvicorn`, launched via `uv run python -m src serve`.
   - Exposes `GET /health`, `GET /search?query=...&k=5`, and `POST /answer` (returning Pydantic-validated `MinimalSearchResults` and `MinimalAnswer` schemas, with interactive Swagger docs at `/docs`).

## Design Decisions
- **SOLID Principles:** Handles (Chunking, Retriever, Generator) are encapsuled and interfaced through the CLI layer.
- **Standardized Libraries:** Adopted industry-standard libraries (`langchain` for splitting, `transformers` for generation) to prevent boilerplate code while keeping the orchestration higly customized.

## Challenges Faced
- **Base Model Limitations:** The 0,6B parameter model tend to hallucinate or enter repetition loops. **Solution:** I implemented a strict chat template and added a `repetition_penalty` to the generation config.
- **Hardware Limitations:** Running inference on CPU for large baches caused OOM (Out Of Memory) kills. **Solution:** I optimized the model loading with `low_cpu_mem_usage=True` and limited the context window.

## Resources
- [BM25 Algorithm Overview](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [vLLM Official Repository](https://github.com/vllm-project/vllm)
- [tqdm Documentation](https://tqdm.github.io/)
- AI was used as a brainstorming partner (ChatGPT/Claude) to refine the custom BM25 tokenizer regex and troubleshoot PyTorch CPU memory limits during batch generation.
