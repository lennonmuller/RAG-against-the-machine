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
        ...