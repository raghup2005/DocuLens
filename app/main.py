from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.multimodal_llm import MultimodalLLM


class RAGPipeline:
    """
    Multimodal Retrieval-Augmented Generation pipeline.

    Flow:

        Query
          ↓
        Dense + BM25
          ↓
        Weighted RRF
          ↓
        Metadata / Modality / Figure Filter
          ↓
        Reranker
          ↓
        Context + Images
          ↓
        Multimodal LLM
          ↓
        Answer
    """

    def __init__(self):

        self.retriever = HybridRetriever(
            dense_weight=0.7,
            bm25_weight=0.3
        )

        self.reranker = Reranker()

        self.llm = MultimodalLLM()

    # --------------------------------
    # Context builder
    # --------------------------------

    def build_context(
        self,
        chunks: list
    ) -> str:

        context_parts = []

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            metadata = chunk["metadata"]

            content_type = metadata.get(
                "content_type",
                "unknown"
            )

            source = metadata.get(
                "source",
                "unknown"
            )

            page = metadata.get(
                "page",
                "unknown"
            )

            figure_number = metadata.get(
                "figure_number"
            )

            figure_info = ""

            if figure_number is not None:
                figure_info = (
                    f"Figure: {figure_number}\n"
                )

            context_parts.append(
                f"""
Context {i}:
Content Type: {content_type}
Source: {source}
Page: {page}
{figure_info}
Content:
{chunk["text"]}
"""
            )

        return "\n".join(context_parts)

    # --------------------------------
    # Ask
    # --------------------------------

    def ask(
        self,
        query: str,
        content_type: str | None = None,
        figure_number: int | None = None
    ) -> dict:
        """
        Execute the complete multimodal RAG pipeline.

        content_type:

            None   → all modalities
            text   → text only
            table  → tables only
            image  → images only

        figure_number:

            None → no figure filtering
            1    → Figure 1 only
            2    → Figure 2 only
            etc.
        """

        # --------------------------------
        # Step 1: Hybrid retrieval
        # --------------------------------

        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=8,
            dense_k=10,
            bm25_k=10,
            source="1706.03762v7.pdf",
            content_type=content_type,
            figure_number=figure_number
        )

        if not retrieved_chunks:

            return {
                "answer": (
                    "No relevant documents were found."
                ),
                "sources": [],
                "chunks": []
            }

        # --------------------------------
        # Debug retrieval
        # --------------------------------

        print("\n" + "=" * 60)
        print("MULTIMODAL HYBRID RETRIEVAL")
        print("=" * 60)

        for chunk in retrieved_chunks:

            metadata = chunk["metadata"]

            print(
                f"ID: {chunk['id']} | "
                f"Type: {metadata.get('content_type')} | "
                f"Page: {metadata.get('page')} | "
                f"Figure: {metadata.get('figure_number')} | "
                f"RRF: {chunk['rrf_score']}"
            )

        # --------------------------------
        # Step 2: Reranking
        # --------------------------------

        reranked_chunks = self.reranker.rerank(
            query=query,
            retrieved_chunks=retrieved_chunks,
            top_k=3
        )

        if not reranked_chunks:

            return {
                "answer": (
                    "No relevant chunks remained "
                    "after reranking."
                ),
                "sources": [],
                "chunks": []
            }

        # --------------------------------
        # Step 3: Collect image paths
        # --------------------------------

        image_paths = []

        for chunk in reranked_chunks:

            metadata = chunk["metadata"]

            if metadata.get(
                "content_type"
            ) == "image":

                image_path = metadata.get(
                    "image_path"
                )

                if image_path:

                    image_paths.append(
                        image_path
                    )

        # Remove duplicate image paths

        image_paths = list(
            dict.fromkeys(
                image_paths
            )
        )

        # --------------------------------
        # Debug image paths
        # --------------------------------

        print("\n" + "=" * 60)
        print("RETRIEVED IMAGE PATHS")
        print("=" * 60)

        if image_paths:

            for image_path in image_paths:

                print(
                    image_path
                )

        else:

            print(
                "No images retrieved."
            )

        # --------------------------------
        # Debug reranking
        # --------------------------------

        print("\n" + "=" * 60)
        print("RERANKED CHUNKS")
        print("=" * 60)

        for chunk in reranked_chunks:

            metadata = chunk["metadata"]

            print(
                f"ID: {chunk['id']} | "
                f"Type: "
                f"{metadata.get('content_type')} | "
                f"Page: "
                f"{metadata.get('page')} | "
                f"Figure: "
                f"{metadata.get('figure_number')} | "
                f"Distance: "
                f"{chunk.get('distance')} | "
                f"Rerank Score: "
                f"{chunk.get('rerank_score')}"
            )

        # --------------------------------
        # Step 4: Build context
        # --------------------------------

        context = self.build_context(
            reranked_chunks
        )

        print("\n" + "=" * 60)
        print("CONTEXT SENT TO LLM")
        print("=" * 60)

        print(context)

        # --------------------------------
        # Step 5: Generate answer
        # --------------------------------

        answer = self.llm.generate(
            query=query,
            context=context,
            image_paths=image_paths
        )

        # --------------------------------
        # Step 6: Sources
        # --------------------------------

        sources = []

        for chunk in reranked_chunks:

            metadata = chunk["metadata"]

            sources.append({
                "source": metadata.get(
                    "source",
                    "unknown"
                ),
                "page": metadata.get(
                    "page",
                    "unknown"
                ),
                "content_type": metadata.get(
                    "content_type",
                    "unknown"
                ),
                "figure_number": metadata.get(
                    "figure_number"
                ),
                "rerank_score": chunk.get(
                    "rerank_score"
                )
            })

        # --------------------------------
        # Final result
        # --------------------------------

        return {
            "answer": answer,
            "sources": sources,
            "chunks": reranked_chunks
        }


# ========================================
# Main
# ========================================

if __name__ == "__main__":

    rag = RAGPipeline()

    # --------------------------------
    # Select modality
    # --------------------------------

    content_type = input(
        "\nSearch type "
        "[all/text/table/image]: "
    ).strip().lower()

    if content_type == "all":

        content_type = None

    elif content_type not in {
        "text",
        "table",
        "image"
    }:

        print(
            "Invalid search type. "
            "Using all modalities."
        )

        content_type = None

    # --------------------------------
    # Figure number
    # --------------------------------

    figure_input = input(
        "\nFigure number "
        "(press Enter to skip): "
    ).strip()

    if figure_input:

        try:

            figure_number = int(
                figure_input
            )

        except ValueError:

            print(
                "Invalid figure number. "
                "Skipping figure filter."
            )

            figure_number = None

    else:

        figure_number = None

    # --------------------------------
    # User query
    # --------------------------------

    query = input(
        "\nEnter your question: "
    ).strip()

    # --------------------------------
    # Run RAG pipeline
    # --------------------------------

    result = rag.ask(
        query=query,
        content_type=content_type,
        figure_number=figure_number
    )

    # --------------------------------
    # Final answer
    # --------------------------------

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(
        result["answer"]
    )

    # --------------------------------
    # Sources
    # --------------------------------

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:

        print(
            f"Source: {source['source']} | "
            f"Page: {source['page']} | "
            f"Type: {source['content_type']} | "
            f"Figure: {source['figure_number']} | "
            f"Rerank Score: "
            f"{source['rerank_score']}"
        )