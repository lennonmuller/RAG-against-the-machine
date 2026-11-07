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

The system provides a CLI built with `fire`.
```bash
# 1. Index the corpus (must place the vLLM corpus in data/raw/)
make run ARGS="index --max_chunk_size 2000"

# 2. Search a single query
make run ARGS="search 'What is vLLM?' --k 5"

# 3. Generate an answer for a single query
make run ARGS="answer 'What hardware platforms does vLLM support?' --k 5"
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
-  **Indexing Time:** ~5 seconds for nearly 2000 files (well below the 5-minute limit).
- **Retrieval Throughput:** Processes 100 queries in 1 < second.
- **Recall@5:** Achieved **83,0%** on the `dataset_docs_public.json`, surpassing the 80% threshold.

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
