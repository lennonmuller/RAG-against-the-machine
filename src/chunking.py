from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    """
    Represents a chunk of text extracted from a file.
    """
    file_path: str
    text: str
    first_character_index: int
    last_character_index: int


class BaseChunker:
    """
    Base class for recursive character chunking
    """

    def __init__(self, splitter: RecursiveCharacterTextSplitter):
        self.splitter = splitter

    def chunk(self, text: str, file_path: str) -> List[DocumentChunk]:
        """
        Splits text using langchain splitter and maps it back to our documentchunk
        """

        doc = Document(
            page_content=text,
            metadatas=[{"file_path": file_path}]
        )
        documents = self.splitter.split_documents([doc])

        chunks = []
        for d in documents:
            start_idx = int(d.metadata.get("start_index", 0))
            end_idx = start_idx + len(d.page_content)

            chunks.append(DocumentChunk(
                file_path=file_path,
                text=d.page_content,
                first_character_index=start_idx,
                last_character_index=end_idx
            ))

        return chunks


class PythonChunker(BaseChunker):
    """
    Chunking stratergy specifically designed for python source code
    using langchains rules
    """

    def __init__(self, max_chunk_size: int = 2000):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
        )
        super().__init__(splitter)


class MarkdownChunker(BaseChunker):
    """
    Chunking strategy specifically designed for Markdown an plain text.
    """
    def __init__(self, max_chunk_size: int = 2000):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
        )
        super().__init__(splitter)
