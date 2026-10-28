import os
import pickle
from pathlib import Path

import List
from tqdm import tqdm

from src.chunking import DocumentChunk, MarkdownChunker, PythonChunker

class ProjectIndexer:
    """
    Handles reading raw files, chunking them and
    persisting the data.
    """
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", max_chunk_size: int = 2000):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.max_chunk_size = max_chunk_size

        self.python_chunker = PythonChunker(max_chunk_size=max_chunk_size)
        self.markdown_chunker = MarkdownChunker(max_chunk_size=max_chunk_size)

        self.processed_dir.mkdir(parents=True, exist_ok=True)


    def _get_valid_files(self) -> List[Path]:
        ... # stopped here.. (implement valid source files .py .md .txt ...)


    def run_ingestion(self) -> None:
        ...
