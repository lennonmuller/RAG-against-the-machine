import json
import pickle
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
        clean_text = text.replace('.', ' ').replace('_', ' ').replace('(', ' ').replace(')', ' ').lower()
        return clean_text.split()

    def search_single(self):
        ...
