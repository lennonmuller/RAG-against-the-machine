import os
import pickle
from pathlib import Path

from typing import List
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
        """
        Recursively finds valid source files (.py, .md, .txt, .rst)
        """
        valid_extensions = {".py", ".md", ".txt", ".rst"}
        valid_files = []
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if Path(file).suffix in valid_extensions:
                    valid_files.append(Path(root) / file)
        return valid_files
    
    def run_ingestion(self) -> None:
        """Reads all files, chunks them, and saves to disk."""
        files = self._get_valid_files()
        print(f"Found {len(files)} files to index in '{self.raw_dir}'.")

        all_chunks: List[DocumentChunk] = []

        for file_path in tqdm(files, desc="Chunking files"):
            try:
                str_path = str(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if file_path.suffix == ".py":
                    chunks = self.python_chunker.chunk(content, str_path)
                else:
                    chunks = self.markdown_chunker.chunk(content, str_path)

                all_chunks.extend(chunks)

            except Exception as e:
                print(f"Warning: Failed to process {file_path}. Reason: {e}")

        chunks_path = self.processed_dir / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(all_chunks, f)

        print(f"Ingestion complete! Indexed {len(all_chunks)} chunks under {self.processed_dir}/")
