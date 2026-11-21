import json
import pickle
import re
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from src.cache import QueryCache
from src.chunking import DocumentChunk
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class BM25Retriever:
    """
    Loads the saved BM25 index and retrieves relevant documents for queries.
    """
    _INDEX_CACHE: dict[str, tuple[Any, Any]] = {}

    def __init__(self, index_path: str = "data/processed/index.pkl"):
        self.index_path = Path(index_path)
        self.chunks: List[DocumentChunk] = []
        self.bm25: Any = None
        self.cache = QueryCache()
        self._load_index()

    def _load_index(self) -> None:
        """Loads the chunks and BM25 model, using in-memory cache if available."""
        key = str(self.index_path.resolve())
        if key in BM25Retriever._INDEX_CACHE:
            self.chunks, self.bm25 = BM25Retriever._INDEX_CACHE[key]
            return
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file {self.index_path} not found. Run index command first")

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.bm25 = data["bm25_model"]
            BM25Retriever._INDEX_CACHE[key] = (self.chunks, self.bm25)

    def _tokenize(self, text: str) -> List[str]:
        """
        Code-aware tokenizer: keeps full identifiers and splits CamelCase/snake_case.
        """
        raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
        tokens: List[str] = []
        for tok in raw_tokens:
            lower_tok = tok.lower()
            tokens.append(lower_tok)
            camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok)
            camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel_split)
            sub_parts = re.split(r"[_\s]+", camel_split.lower())
            if len(sub_parts) > 1:
                tokens.extend(p for p in sub_parts if p and p != lower_tok)
        return tokens

    def search_single(self, query_id: str, query: str, k: int) -> MinimalSearchResults:
        """
        Runs BM25 on a single query and maps the results to Pydantic MinimalSearchResults.
        """
        query_str = str(query).strip()
        if k <= 0 or not query_str:
            return MinimalSearchResults(
                question_id=query_id,
                question=query_str,
                retrieved_sources=[],
            )

        cached = self.cache.get("search", query_str, k)
        if cached is not None:
            return MinimalSearchResults(
                question_id=query_id,
                question=query_str,
                retrieved_sources=[MinimalSource.model_validate(s) for s in cached],
            )

        tokenized_query = self._tokenize(query_str)
        if not tokenized_query:
            return MinimalSearchResults(
                question_id=query_id,
                question=query_str,
                retrieved_sources=[],
            )

        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        retrieved_sources = []
        for idx in top_k_indices:
            chunk = self.chunks[idx]
            retrieved_sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )

        self.cache.set(
            "search",
            query_str,
            k,
            [s.model_dump() for s in retrieved_sources],
        )

        return MinimalSearchResults(
            question_id=query_id,
            question=query_str,
            retrieved_sources=retrieved_sources,
        )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """
        Runs batch retrieval on a full dataset and dumps the exact JSON schema required.
        """
        if k <= 0:
            print("Error: k must be greater than 0.")
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
            dataset = RagDataset.model_validate(raw_json)
        except FileNotFoundError:
            print(f"Error: Dataset file not found: '{dataset_path}'")
            return
        except Exception as e:
            print(f"Error: Failed to load or validate dataset '{dataset_path}': {e}")
            return

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / Path(dataset_path).name

        results_list = []
        for q in tqdm(dataset.rag_questions, desc="Searching queries"):
            res = self.search_single(
                query_id=q.question_id,
                query=q.question,
                k=k,
            )
            results_list.append(res)

        final_output = StudentSearchResults(
            search_results=results_list,
            k=k,
        )

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_output.model_dump_json(indent=2))

        print(f"Saved student_search_results to {out_file}")
