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
