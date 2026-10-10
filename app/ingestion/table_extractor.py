import os
import csv
import pymupdf


def extract_tables_from_pdf(
    file_path: str,
    output_dir: str = "data/processed/tables"
) -> list:
    """
    Extract tables from a PDF and save them as CSV files.

    Args:
        file_path: Path to the PDF file.
        output_dir: Directory where extracted tables are stored.

    Returns:
        List of paths to extracted CSV files.
    """

    os.makedirs(output_dir, exist_ok=True)

    document = pymupdf.open(file_path)

    extracted_tables = []

    for page_number, page in enumerate(document):

        try:
            tables = page.find_tables()
        except Exception as error:
            print(
                f"Could not detect tables on page "
                f"{page_number + 1}: {error}"
            )
            continue

        if not tables.tables:
            continue

        for table_number, table in enumerate(tables.tables):

            table_data = table.extract()

            if not table_data:
                continue

            file_name = (
                f"page_{page_number + 1}"
                f"_table_{table_number + 1}.csv"
            )

            output_path = os.path.join(
                output_dir,
                file_name
            )

            with open(
                output_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                writer = csv.writer(csv_file)
                writer.writerows(table_data)

            extracted_tables.append(output_path)

    document.close()

    return extracted_tables


if __name__ == "__main__":

    file_path = "data/raw/1706.03762v7.pdf"

    tables = extract_tables_from_pdf(file_path)

    print(f"Extracted {len(tables)} tables.")

    for table in tables:
        print(table)