import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

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
    def __init__(self, index_path: str = "data/processed/index.pkl"):
        self.index_path = Path(index_path)
        self.chunks = []
        self.bm25 = None
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file {self.index_path} not found. Run index command first")

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.bm25 = data["bm25_model"]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-z0-9]+', text.lower())

    def search_single(self, query_id: str, query: str, k: int) -> MinimalSearchResults:
        """
        Search single minimal results based on recall@k.
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        retrieved_sources = []
        for idx in top_k_indices:
            chunk = self.chunks[idx]

            source = MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index
            )
            retrieved_sources.append(source)

        return MinimalSearchResults(
            question_id=query_id,
            question=query,
            retrieved_sources=retrieved_sources
        )

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """
        Runs batch retrieval on a full dataset and dumps the JSON schema required.
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(dataset_path).name
        out_file = out_dir / dataset_name

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        dataset = RagDataset.model_validate(raw_json)

        results_list = []
        for q in tqdm(dataset.rag_questions, desc="Searching queries"):
            res = self.search_single(
                query_id=q.question_id,
                query=q.question,
                k=k
            )
            results_list.append(res)

        final_output = StudentSearchResults(
            search_results=results_list,
            k=k
        )

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_output.model_dump_json(indent=2))

        print(f"Saved student_search_results to {out_file}")
