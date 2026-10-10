import re
import nltk

from nltk.tokenize import sent_tokenize


# ============================================================
# 1. FIXED-SIZE CHUNKING
# ============================================================

def fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Split text into fixed-size chunks using characters.

    Args:
        text: Input text.
        chunk_size: Maximum characters in each chunk.
        overlap: Number of overlapping characters.

    Returns:
        List of text chunks.
    """

    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ============================================================
# 2. SENTENCE-BASED CHUNKING
# ============================================================

def sentence_chunking(
    text: str,
    sentences_per_chunk: int = 5
) -> list[str]:
    """
    Group a fixed number of sentences into chunks.
    """

    if not text.strip():
        return []

    sentences = sent_tokenize(text)

    chunks = []

    for i in range(
        0,
        len(sentences),
        sentences_per_chunk
    ):

        chunk = " ".join(
            sentences[i:i + sentences_per_chunk]
        )

        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


# ============================================================
# 3. PARAGRAPH-BASED CHUNKING
# ============================================================

def paragraph_chunking(
    text: str,
    max_characters: int = 1500
) -> list[str]:
    """
    Split text based on paragraphs while keeping
    chunks below a maximum character size.
    """

    if not text.strip():
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        if not current_chunk:

            current_chunk = paragraph

        elif (
            len(current_chunk)
            + len(paragraph)
            + 2
            <= max_characters
        ):

            current_chunk += "\n\n" + paragraph

        else:

            chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ============================================================
# 4. RECURSIVE CHUNKING
# ============================================================

def recursive_chunking(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100
) -> list[str]:
    """
    Recursively split text using natural separators.

    Separators:
        1. Paragraph
        2. Line
        3. Sentence
        4. Space
    """

    separators = [
        "\n\n",
        "\n",
        ". ",
        " "
    ]

    return _recursive_split(
        text,
        separators,
        chunk_size,
        overlap
    )


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int
) -> list[str]:

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return fixed_size_chunking(
            text,
            chunk_size,
            overlap
        )

    separator = separators[0]

    parts = text.split(separator)

    if len(parts) == 1:

        return _recursive_split(
            text,
            separators[1:],
            chunk_size,
            overlap
        )

    chunks = []

    current_chunk = ""

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if (
            len(current_chunk)
            + len(part)
            + len(separator)
            <= chunk_size
        ):

            if current_chunk:
                current_chunk += separator

            current_chunk += part

        else:

            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return _apply_overlap(
        chunks,
        overlap
    )


def _apply_overlap(
    chunks: list[str],
    overlap: int
) -> list[str]:

    if not chunks or overlap <= 0:
        return chunks

    overlapped_chunks = [chunks[0]]

    for i in range(1, len(chunks)):

        previous_chunk = chunks[i - 1]

        overlap_text = previous_chunk[-overlap:]

        new_chunk = (
            overlap_text
            + " "
            + chunks[i]
        )

        overlapped_chunks.append(
            new_chunk.strip()
        )

    return overlapped_chunks

if __name__ == "__main__":

    sample_text = """
    Transformers are neural network architectures.

    They use attention mechanisms to process information.
    Self-attention allows each token to interact with
    other tokens in the sequence.

    Transformers are widely used in modern NLP systems.
    They are also used in large language models.
    """

    print("\n===== FIXED SIZE =====")

    chunks = fixed_size_chunking(
        sample_text,
        chunk_size=100,
        overlap=20
    )

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)


    print("\n===== SENTENCE =====")

    chunks = sentence_chunking(
        sample_text,
        sentences_per_chunk=2
    )

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)


    print("\n===== PARAGRAPH =====")

    chunks = paragraph_chunking(
        sample_text,
        max_characters=150
    )

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)


    print("\n===== RECURSIVE =====")

    chunks = recursive_chunking(
        sample_text,
        chunk_size=150,
        overlap=30
    )

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)