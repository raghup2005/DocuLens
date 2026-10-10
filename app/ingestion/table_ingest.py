import csv
import re
from pathlib import Path

from app.ingestion.table_extractor import (
    extract_tables_from_pdf
)

from app.vectorstore import VectorStore


def table_to_text(
    table_rows: list[list]
) -> str:
    """
    Convert extracted table rows into searchable text.
    """

    if not table_rows:
        return ""

    rows = []

    for row in table_rows:

        cleaned_cells = []

        for cell in row:

            if cell is None:
                cell = ""

            cell = str(cell).strip()

            cleaned_cells.append(cell)

        row_text = " | ".join(
            cleaned_cells
        )

        if row_text.strip():
            rows.append(row_text)

    return "\n".join(rows)


def extract_page_and_table_number(
    file_path: str
) -> tuple[int, int]:
    """
    Extract page number and table number from
    filenames such as:

        page_9_table_1.csv
    """

    file_name = Path(
        file_path
    ).name

    match = re.search(
        r"page_(\d+)_table_(\d+)",
        file_name
    )

    if not match:
        raise ValueError(
            f"Could not determine page/table number "
            f"from filename: {file_name}"
        )

    page_number = int(
        match.group(1)
    )

    table_number = int(
        match.group(2)
    )

    return page_number, table_number


def ingest_tables(
    pdf_path: str
) -> None:
    """
    Extract tables from a PDF and store them
    as searchable records in ChromaDB.
    """

    print("\n" + "=" * 60)
    print("TABLE INGESTION")
    print("=" * 60)

    # --------------------------------
    # Step 1: Extract tables
    # --------------------------------

    print("\nExtracting tables...")

    table_files = extract_tables_from_pdf(
        pdf_path
    )

    print(
        "Tables extracted:",
        len(table_files)
    )

    if not table_files:
        print(
            "\nNo tables were detected."
        )

        return

    # --------------------------------
    # Step 2: Prepare records
    # --------------------------------

    documents = []
    metadatas = []
    ids = []

    source_name = Path(
        pdf_path
    ).name

    source_stem = Path(
        pdf_path
    ).stem

    for table_file in table_files:

        # ----------------------------
        # Get page/table number
        # ----------------------------

        page_number, table_number = (
            extract_page_and_table_number(
                table_file
            )
        )

        # ----------------------------
        # Read CSV
        # ----------------------------

        with open(
            table_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            reader = csv.reader(
                csv_file
            )

            table_rows = list(reader)

        # ----------------------------
        # Convert table to text
        # ----------------------------

        table_text = table_to_text(
            table_rows
        )

        if not table_text.strip():
            continue

        # Add useful context
        searchable_text = (
            f"Table from {source_name}. "
            f"Page {page_number}. "
            f"Table {table_number}.\n\n"
            f"{table_text}"
        )

        documents.append(
            searchable_text
        )

        metadatas.append({
            "source": source_name,
            "page": page_number,
            "content_type": "table",
            "table_index": table_number
        })

        ids.append(
            f"{source_stem}"
            f"_page_{page_number}"
            f"_table_{table_number}"
        )

    # --------------------------------
    # Step 3: Store in ChromaDB
    # --------------------------------

    if not documents:

        print(
            "\nNo searchable table data was created."
        )

        return

    print(
        "\nStoring tables in ChromaDB..."
    )

    vector_store = VectorStore()

    vector_store.add_documents(
        texts=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(
        "\nTables stored:",
        len(documents)
    )

    print(
        "Total records in ChromaDB:",
        vector_store.count()
    )

    print(
        "\nTable ingestion completed successfully."
    )


if __name__ == "__main__":

    pdf_path = (
        "data/raw/1706.03762v7.pdf"
    )

    ingest_tables(
        pdf_path
    )