from typing import List

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    chunk_size: max characters per chunk
    chunk_overlap: how many characters to repeat between chunks (gives the LLM context continuity)
    """
    if not text or not text.strip():
        return []

    # Clean up whitespace
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to break at a sentence or word boundary
        if end < len(text):
            # Try to break at sentence boundary (. ! ?)
            for punct in ['. ', '! ', '? ']:
                boundary = text.rfind(punct, start, end)
                if boundary != -1:
                    end = boundary + 1
                    break
            else:
                # Fall back to word boundary
                boundary = text.rfind(' ', start, end)
                if boundary != -1:
                    end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, but overlap with previous chunk
        start = end - chunk_overlap
        if start >= len(text):
            break

    return chunks
