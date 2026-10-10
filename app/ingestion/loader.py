from pathlib import Path

from app.ingestion.file_detector import detect_file_type


def load_file(file_path: str) -> dict:
    """
    Load a file and return basic information about it.

    Args:
        file_path: Path to the input file.

    Returns:
        Dictionary containing file path, name, extension,
        file type, and file size.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    file_type = detect_file_type(file_path)

    file_info = {
        "file_path": str(path),
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "file_type": file_type,
        "file_size": path.stat().st_size,
    }

    return file_info