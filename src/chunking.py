import re
from dataclasses import dataclass

import List
import Tuple


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

    def _recursive_chunk(self, text: str, absolute_start: int, file_path: str, separators: List[str]) -> List[DocumentChunk]:
        """
        Recursively splits text using a list of separator patterns until chunks fit max_chunk_size.
        """
        if len(text) <= self.max_chunk_size or not separators:
            return [DocumentChunk(
                file_path=file_path,
                text=text,
                first_character_index=absolute_start,
                last_character_index=absolute_start + len(text)
            )]

        current_separator = separators[0]
        next_separators = separators[1:]

        segments = self._split_text_by_pattern(text, current_separator)

        chunks = []
        current_chunk_text = ""
        current_chunk_start = absolute_start

        for local_start, local_end, segment in segments:
            segment_len = len(segment)

            # If a single segment is too big, recurse on it
            if segment_len > self.max_chunk_size:
                if current_chunk_text:
                    chunks.append(DocumentChunk(
                        file_path=file_path,
                        text=current_chunk_text,
                        first_character_index=current_chunk_start,
                        last_character_index=current_chunk_start + len(current_chunk_text)
                    ))  # first, yield whateve accumulated
                    current_chunk_text = ""

                # Then recurse into the oversized segment
                sub_chunks = self._recursive_chunk(
                    text=segment,
                    absolute_start=absolute_start + local_start,
                    file_path=file_path,
                    separators=next_separators
                )
                chunks.extend(sub_chunks)
                # Update current start for the next iteration
                current_chunk_start = absolute_start + local_end
                continue

            # If adding this segment exceeds max size, yield the current chunk
            if len(current_chunk_text) + segment_len > self.max_chunk_size:
                chunks.append(DocumentChunk(
                    file_path=file_path,
                    text=current_chunk_text,
                    first_character_index=current_chunk_start,
                    last_character_index=current_chunk_start + len(current_chunk_text)
                ))
                current_chunk_text = segment
                current_chunk_start = absolute_start + local_start
            else:
                if not current_chunk_text:
                    current_chunk_start = absolute_start + local_start
                current_chunk_text += segment


        if current_chunk_text:
            chunks.append(DocumentChunk(
                file_path=file_path,
                text=current_chunk_text,
                first_character_index=current_chunk_start,
                last_character_index=current_chunk_start + len(current_chunk_text)
            ))

        return chunks


class PythonChunker(BaseChunker):
    """
    Chunking stratergy specifically designed for python source code
    """

    def chunk(self, text: str, file_path: str) -> List[DocumentChunk]:
        # split by top-level class or function def, then paragraphs, then lines
        separators = [
            r"(?=\n(?:class|def|async def)\s)",
            r"(?=\n\n)",
            r"(?=\n)"
        ]
        return self._recursive_chunk(text, 0, file_path, separators)


class MarkdownChunker(BaseChunker):
    """
    Chunking strategy specifically designed for Markdown an plain text.
    """
    def chunk(self, text: str, file_path: str) -> List[DocumentChunk]:
        # split by headers, then paragraphers, then lines
        separators = [
            r"(?=\n#+\s)",
            r"(?=\n\n)",
            r"(?=\n)"
        ]
        return self._recursive_chunk(text, 0, file_path, separators)
