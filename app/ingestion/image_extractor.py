import os
import pymupdf

def extract_images_from_pdf(
    file_path: str,
    output_dir: str = "data/images"
) -> list:
    """
    Extract images from a PDF and save them to the output directory.

    Args:
        file_path: Path to the PDF file.
        output_dir: Directory where extracted images will be stored.

    Returns:
        List of paths of extracted images.
    """

    os.makedirs(output_dir, exist_ok=True)

    document = pymupdf.open(file_path)

    extracted_images = []

    for page_number, page in enumerate(document):

        images = page.get_images(full=True)

        for image_number, image in enumerate(images):

            xref = image[0]

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]
            image_extension = image_data["ext"]

            image_name = (
                f"page_{page_number + 1}"
                f"_image_{image_number + 1}"
                f".{image_extension}"
            )

            image_path = os.path.join(
                output_dir,
                image_name
            )

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            extracted_images.append(image_path)

    document.close()

    return extracted_images

if __name__ == "__main__":

    file_path = "data/raw/1706.03762v7.pdf"

    images = extract_images_from_pdf(file_path)

    print(f"Extracted {len(images)} images.")

    for image in images:
        print(image)