from pathlib import Path

import ollama


class MultimodalLLM:
    """
    Multimodal LLM used for answering questions using
    retrieved text and image context.
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:3b"
    ):
        self.model = model

    def generate(
        self,
        query: str,
        context: str,
        image_paths: list[str] | None = None
    ) -> str:

        image_paths = image_paths or []

        prompt = f"""
You are answering a question using retrieved information from a technical research paper.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

IMPORTANT INSTRUCTIONS:

1. Use ALL relevant retrieved contexts.
2. If multiple images belong to the same figure, consider all of them together.
3. Do NOT answer using only the first or highest-ranked context.
4. Pay attention to "Panel Position" such as LEFT and RIGHT.
5. If the question asks "what does Figure X contain/show?", describe the complete figure, including all retrieved panels.
6. Clearly distinguish different panels when necessary.
7. Use only information supported by the retrieved context and visible image information.
8. Do not invent missing details.
9. Do not mention RAG, multimodal RAG, NLP, computer vision, or the retrieval system unless that information is explicitly present in the retrieved source.
10. Give a concise but complete answer.

For a multi-panel figure, structure the answer like:

Figure X contains:
- Left panel: ...
- Right panel: ...

Then briefly explain what the figure represents as a whole.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_paths
                }
            ],
            options={
                "num_ctx": 8192,
                "temperature": 0
            }
        )

        return response["message"]["content"]

# ====================================
# TEST
# ====================================

if __name__ == "__main__":

    llm = MultimodalLLM()

    context = """
Context 1:
Content Type: image
Source: 1706.03762v7.pdf
Page: 4

The image contains Q, K, V, MatMul,
Scale, Mask, and SoftMax operations.
"""

    image_path = (
        "data/images/page_4_image_1.png"
    )

    answer = llm.generate(
        query="What does this attention diagram show?",
        context=context,
        image_paths=[image_path]
    )

    print("\nGenerated Answer:\n")
    print(answer)