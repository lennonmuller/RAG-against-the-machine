import re
import List, Tuple
from dataclasses import dataclass


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

    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def _split_text_by_pattern(self, text: str, pattern: str) -> List[Tuple[int, int, str]]:
        """
        Splits text by a regex pattern, returning a list of
        (start_idx, end_idx, segment).
        Uses lookarounds to avoid consuming the split characters
        """

        segments = []
        last_end = 0

        for match in re.finditer(pattern, text):
            start = match.start()
            if start > last_end:
                segments.append((last_end, start, text[last_end:start]))
            last_end = start

        if last_end < len(text):
            segments.append((last_end, len(text), text[last_end:]))

        return segments