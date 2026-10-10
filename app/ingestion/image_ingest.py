import re
from pathlib import Path

import ollama
import pymupdf

from app.ingestion.image_extractor import (
    extract_images_from_pdf
)

from app.vectorstore import VectorStore


class ImageIngestor:
    """
    Extract images from a PDF, extract nearby figure captions,
    generate accurate vision descriptions, and store searchable
    image representations in ChromaDB.
    """

    def __init__(
        self,
        vision_model: str = "qwen2.5vl:3b"
    ):
        self.vision_model = vision_model
        self.vector_store = VectorStore()

    # --------------------------------
    # Extract page/image information
    # --------------------------------

    def get_image_metadata(
        self,
        image_path: str
    ) -> tuple[int, int]:
        """
        Extract page number and image number from:

            page_3_image_1.png
        """

        file_name = Path(
            image_path
        ).name

        match = re.search(
            r"page_(\d+)_image_(\d+)",
            file_name
        )

        if not match:

            raise ValueError(
                f"Could not determine page/image "
                f"number from: {file_name}"
            )

        page_number = int(
            match.group(1)
        )

        image_number = int(
            match.group(2)
        )

        return page_number, image_number

    # --------------------------------
    # Extract figure caption
    # --------------------------------

    def extract_figure_caption(
        self,
        pdf_path: str,
        page_number: int
    ) -> str:
        """
        Extract a complete figure caption from a PDF page.

        PDF captions are often wrapped across multiple lines.
        This keeps the wrapped caption together.
        """

        document = pymupdf.open(
            pdf_path
        )

        try:

            page = document[
                page_number - 1
            ]

            page_text = page.get_text(
                "text"
            )

        finally:

            document.close()

        # --------------------------------
        # Find figure caption
        # --------------------------------

        pattern = re.compile(
            r"((?:Figure|Fig\.)\s*\d+\s*:.*?)(?=\n\s*\n|\Z)",
            re.IGNORECASE | re.DOTALL
        )

        matches = pattern.findall(
            page_text
        )

        if not matches:

            return ""

        # --------------------------------
        # Clean wrapped caption
        # --------------------------------

        caption = matches[0].strip()

        caption = re.sub(
            r"\s+",
            " ",
            caption
        )

        return caption


    # --------------------------------
    # Determine panel information
    # --------------------------------

    def get_panel_hint(
        self,
        caption: str,
        image_number: int | None
    ) -> str:
        """
        Determine panel position only when the caption explicitly
        describes a LEFT/RIGHT multi-panel figure.

        For a two-panel figure with (left) and (right) in its caption:
            image_1 -> LEFT panel
            image_2 -> RIGHT panel

        This mapping is NOT applied to unrelated figures.
        """

        if not caption or image_number is None:
            return ""

        caption_lower = caption.lower()

        has_left = "(left)" in caption_lower
        has_right = "(right)" in caption_lower

        if not (has_left and has_right):
            return ""

        if image_number == 1:
            return """
Panel position:
The figure caption explicitly identifies LEFT and RIGHT panels.
This is the FIRST extracted image from the figure.

For this two-panel figure, treat this image as the LEFT panel.
Describe only the LEFT panel shown in the supplied image.
"""

        if image_number == 2:
            return """
Panel position:
The figure caption explicitly identifies LEFT and RIGHT panels.
This is the SECOND extracted image from the figure.

For this two-panel figure, treat this image as the RIGHT panel.
Describe only the RIGHT panel shown in the supplied image.
"""

        return ""


    # --------------------------------
    # Vision description
    # --------------------------------

    def describe_image(
        self,
        image_path: str,
        caption: str = "",
        figure_number: int | None = None,
        image_number: int | None = None
    ) -> str:
        """
        Generate an accurate technical description
        of an image or figure panel.
        """

        # --------------------------------
        # Determine panel hint
        # --------------------------------

        panel_hint = self.get_panel_hint(
            caption=caption,
            image_number=image_number
        )

        # --------------------------------
        # Figure context
        # --------------------------------

        figure_context = ""

        if figure_number is not None:

            figure_context += (
                f"This image belongs to Figure "
                f"{figure_number}.\n"
            )

        if image_number is not None:

            figure_context += (
                f"Extracted image number: "
                f"{image_number}.\n"
            )

        if caption:

            figure_context += (
                f"\nFigure caption:\n"
                f"{caption}\n"
            )

        # --------------------------------
        # Vision prompt
        # --------------------------------

        prompt = f"""
You are a vision model analyzing an image extracted from
a technical research paper.

{figure_context}

{panel_hint}

Your task is to create a factual description of ONLY the
content visible in the supplied image.

IMPORTANT PANEL RULES:

1. If a panel position is explicitly supplied above, follow it.
2. Do NOT change the panel identity based on your own guess.
3. If the supplied image contains only one panel, describe
   only that panel.
4. Do NOT describe a different panel from the same figure.
5. Do NOT assume that image number always means left or right.
   Use a left/right mapping only when the figure caption explicitly
   supports that mapping.

IMPORTANT CAPTION RULE:

The figure caption may contain information about multiple panels.
Use the caption only to identify the subject of the current panel
when that information is explicitly stated.

For example, if the caption says:
(left) Scaled Dot-Product Attention
(right) Multi-Head Attention

and the supplied panel is identified as the RIGHT panel,
the main subject should be Multi-Head Attention.

IMPORTANT FACTUAL RULES:

- Use ONLY information visible in the image or explicitly supported
  by the figure caption.
- Do NOT add outside background knowledge.
- Do NOT mention RAG, multimodal RAG, NLP, computer vision,
  transformers, or any other broader system unless it is explicitly
  visible or stated in the image/caption.
- Do NOT infer a component's purpose merely from background knowledge.
- Do NOT invent equations.
- Do NOT invent labels.
- Do NOT invent connections or data flow.
- Do NOT reconstruct parts that are not visible.
- Preserve visible labels exactly when possible.
- If something is unclear, say that it is unclear instead of guessing.

FIGURE CONTEXT:

{caption if caption else "No figure caption available."}

Describe the image using exactly these sections:

1. Panel identification
2. Main subject
3. Important visible labels
4. Main components
5. Visible connections and flow
6. Visible equations or symbols
7. Technical information directly supported by the image

For Panel identification:
- State LEFT panel, RIGHT panel, or UNKNOWN/UNSPECIFIED.
- Include the reason only when it is explicitly supported.

For Main subject:
- Prefer the subject explicitly identified by the figure caption.
- Otherwise identify only what is clearly visible.

Be precise, concise, and factual.
"""

        # --------------------------------
        # Ollama vision model
        # --------------------------------

        response = ollama.chat(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        image_path
                    ]
                }
            ],
            options={
                "num_ctx": 8192,
                "temperature": 0
            }
        )

        return response[
            "message"
        ]["content"].strip()


    # --------------------------------
    # Complete image ingestion
    # --------------------------------

    def ingest(
        self,
        pdf_path: str
    ) -> None:
        """
        Extract images, captions, and descriptions,
        then store them in ChromaDB.
        """

        print("\n" + "=" * 60)
        print("IMAGE INGESTION")
        print("=" * 60)

        # --------------------------------
        # Extract images
        # --------------------------------

        print("\nExtracting images...")

        image_paths = extract_images_from_pdf(
            pdf_path
        )

        print(
            "Images extracted:",
            len(image_paths)
        )

        if not image_paths:

            print(
                "\nNo raster images were extracted."
            )

            return

        documents = []
        metadatas = []
        ids = []

        source_name = Path(
            pdf_path
        ).name

        source_stem = Path(
            pdf_path
        ).stem

        # --------------------------------
        # Process images
        # --------------------------------

        for image_path in image_paths:

            print(
                "\nProcessing:",
                image_path
            )

            try:

                # ----------------------------
                # Page/image metadata
                # ----------------------------

                page_number, image_number = (
                    self.get_image_metadata(
                        image_path
                    )
                )

                # ----------------------------
                # Extract figure caption
                # ----------------------------

                caption = (
                    self.extract_figure_caption(
                        pdf_path,
                        page_number
                    )
                )

                if caption:

                    print(
                        "Caption found:",
                        caption
                    )

                else:

                    print(
                        "No figure caption found."
                    )

                # ----------------------------
                # Extract figure number
                # ----------------------------

                figure_number = None

                if caption:

                    figure_match = re.search(
                        r"(?:Figure|Fig\.)\s*(\d+)",
                        caption,
                        re.IGNORECASE
                    )

                    if figure_match:

                        figure_number = int(
                            figure_match.group(1)
                        )

                # ----------------------------
                # Vision description
                # ----------------------------

                description = (
                    self.describe_image(
                        image_path=image_path,
                        caption=caption,
                        figure_number=figure_number,
                        image_number=image_number
                    )
                )

                if not description:

                    print(
                        "No description generated."
                    )

                    continue

                print(
                    "Description generated."
                )

                # ----------------------------
                # Searchable text
                # ----------------------------

                searchable_text = (
                    f"Image from {source_name}.\n"
                    f"Page: {page_number}.\n"
                    f"Image: {image_number}.\n"
                )

                if (
                    caption
                    and "(left)" in caption.lower()
                    and "(right)" in caption.lower()
                ):
                    if image_number == 1:
                        searchable_text += (
                            "Panel Position: Left.\n"
                        )
                    elif image_number == 2:
                        searchable_text += (
                            "Panel Position: Right.\n"
                        )

                if figure_number is not None:

                    searchable_text += (
                        f"Figure Number: "
                        f"{figure_number}.\n"
                    )

                if caption:

                    searchable_text += (
                        f"Figure Caption:\n"
                        f"{caption}\n\n"
                    )

                searchable_text += (
                    f"Vision Description:\n"
                    f"{description}"
                )

                documents.append(
                    searchable_text
                )

                # ----------------------------
                # Metadata
                # ----------------------------

                metadata = {
                    "source": source_name,
                    "page": page_number,
                    "content_type": "image",
                    "image_index": image_number,
                    "image_path": str(
                        image_path
                    )
                }

                if (
                    caption
                    and "(left)" in caption.lower()
                    and "(right)" in caption.lower()
                ):
                    if image_number == 1:
                        metadata["panel_position"] = "left"
                    elif image_number == 2:
                        metadata["panel_position"] = "right"

                if figure_number is not None:

                    metadata[
                        "figure_number"
                    ] = figure_number

                metadatas.append(
                    metadata
                )

                # ----------------------------
                # Unique ID
                # ----------------------------

                ids.append(
                    f"{source_stem}"
                    f"_page_{page_number}"
                    f"_image_{image_number}"
                )

            except Exception as error:

                print(
                    "Could not process image:"
                )

                print(
                    image_path
                )

                print(
                    "Error:",
                    error
                )

                print(
                    "Skipping this image..."
                )

        # --------------------------------
        # Store
        # --------------------------------

        if not documents:

            print(
                "\nNo searchable image "
                "records were created."
            )

            return

        print(
            "\nGenerating embeddings "
            "and storing..."
        )

        self.vector_store.upsert_documents(
            texts=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(
            "\nImages stored:",
            len(documents)
        )

        print(
            "Total records in ChromaDB:",
            self.vector_store.count()
        )

        print(
            "\nImage ingestion completed."
        )


# ========================================
# Main
# ========================================

if __name__ == "__main__":

    pdf_path = (
        "data/raw/1706.03762v7.pdf"
    )

    ingestor = ImageIngestor()

    ingestor.ingest(
        pdf_path
    )