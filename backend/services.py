from pathlib import Path

from app.ingestion.loader import load_file
from app.ingestion.parser import parse_document
from app.ingestion.chunker import recursive_chunking
from app.vectorstore import VectorStore
from app.ingestion.image_ingest import ImageIngestor


# ---------------------------------------------------------
# Initialize components
# ---------------------------------------------------------

vectorstore = VectorStore()
image_ingestor = ImageIngestor()


# ---------------------------------------------------------
# Text ingestion
# ---------------------------------------------------------

def ingest_text_document(file_path: str, file_info: dict):

    # Parse document
    parsed = parse_document(
        file_path,
        file_info["file_type"]
    )

    # Extract text
    if isinstance(parsed, dict):

        text = parsed.get("text", "")

    elif isinstance(parsed, str):

        text = parsed

    else:

        text = str(parsed)

    if not text.strip():
        raise ValueError(
            "No text could be extracted from the document."
        )

    print(f"Extracted characters: {len(text)}")

    # Chunking
    chunks = recursive_chunking(
        text,
        chunk_size=1000,
        overlap=100
    )

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    print(f"Chunks created: {len(chunks)}")

    # Metadata
    metadatas = []

    for i in range(len(chunks)):

        metadatas.append({
            "source": file_info["file_name"],
            "content_type": "text",
            "chunk_id": i
        })

    # IDs
    document_name = Path(file_path).stem

    ids = [
        f"{document_name}_chunk_{i}"
        for i in range(len(chunks))
    ]

    # Store
    vectorstore.add_documents(
        texts=chunks,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Text records stored: {len(chunks)}")

    return len(chunks)


# ---------------------------------------------------------
# Complete multimodal ingestion
# ---------------------------------------------------------

def ingest_document(file_path: str):

    print("\n" + "=" * 60)
    print("MULTIMODAL DOCUMENT INGESTION")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Detect file
    # -----------------------------------------------------

    file_info = load_file(file_path)

    print(f"File: {file_info['file_name']}")
    print(f"Type: {file_info['file_type']}")

    # -----------------------------------------------------
    # 2. Text ingestion
    # -----------------------------------------------------

    text_chunks = ingest_text_document(
        file_path,
        file_info
    )

    # -----------------------------------------------------
    # 3. Image ingestion
    # -----------------------------------------------------

    image_count = 0

    if file_info["file_type"] == "pdf":

        print("\nStarting multimodal image ingestion...")

        before_count = vectorstore.count()

        image_ingestor.ingest(file_path)

        after_count = vectorstore.count()

        # Difference gives newly added/upserted records
        image_count = max(
            after_count - before_count,
            0
        )

    # -----------------------------------------------------
    # 4. Return result
    # -----------------------------------------------------

    return {
        "file_name": file_info["file_name"],
        "file_type": file_info["file_type"],
        "text_chunks": text_chunks,
        "image_records": image_count,
        "status": "success"
    }