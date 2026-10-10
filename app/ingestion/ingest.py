from pathlib import Path

from app.ingestion.loader import load_file
from app.ingestion.parser import parse_pdf_by_page
from app.ingestion.chunker import recursive_chunking
from app.vectorstore import VectorStore


def ingest_pdf(
    file_path: str
) -> None:
    """
    Page-aware PDF ingestion pipeline.

    Flow:

        PDF
        ↓
        Page extraction
        ↓
        Page-level chunking
        ↓
        Embeddings
        ↓
        ChromaDB
    """

    print("\n" + "=" * 60)
    print("PAGE-AWARE PDF INGESTION")
    print("=" * 60)

    # --------------------------------
    # Step 1: Check file
    # --------------------------------

    file_info = load_file(file_path)

    print("\nFile:", file_info["file_name"])
    print("Type:", file_info["file_type"])

    # --------------------------------
    # Step 2: Extract pages
    # --------------------------------

    print("\nExtracting pages...")

    pages = parse_pdf_by_page(
        file_path
    )

    print(
        "Pages with text:",
        len(pages)
    )

    # --------------------------------
    # Step 3: Create chunks
    # --------------------------------

    all_chunks = []
    all_metadatas = []
    all_ids = []

    document_name = Path(file_path).name
    document_stem = Path(file_path).stem

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        chunks = recursive_chunking(
            page_text,
            chunk_size=1000,
            overlap=100
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            all_chunks.append(chunk)

            all_metadatas.append({
                "source": document_name,
                "page": page_number,
                "content_type": "text",
                "chunk_index": chunk_index
            })

            all_ids.append(
                f"{document_stem}"
                f"_page_{page_number}"
                f"_chunk_{chunk_index}"
            )

    print(
        "Total chunks:",
        len(all_chunks)
    )

    if not all_chunks:
        raise ValueError(
            "No chunks were created."
        )

    # --------------------------------
    # Step 4: Store in ChromaDB
    # --------------------------------

    print("\nCreating embeddings...")

    vector_store = VectorStore()

    vector_store.add_documents(
        texts=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print(
        "\nDocuments in ChromaDB:",
        vector_store.count()
    )

    print("\nIngestion completed successfully.")


if __name__ == "__main__":

    pdf_path = (
        "data/raw/1706.03762v7.pdf"
    )

    ingest_pdf(pdf_path)