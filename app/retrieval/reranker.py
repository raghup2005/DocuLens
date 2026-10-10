
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved document chunks
    using a Cross-Encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        retrieved_chunks: list,
        top_k: int = 3
    ) -> list:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not retrieved_chunks:
            return []

        # Create query-document pairs
        pairs = [
            (query, chunk["text"])
            for chunk in retrieved_chunks
        ]

        # Calculate relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to retrieved chunks
        reranked_results = []

        for chunk, score in zip(retrieved_chunks, scores):

            result = chunk.copy()

            result["rerank_score"] = float(score)

            reranked_results.append(result)

        # Sort by relevance score (highest first)
        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # Return top K results
        return reranked_results[:top_k]


if __name__ == "__main__":

    from app.retrieval.retriever import Retriever

    query = "How does self-attention work?"

    # Step 1: Retrieve candidate chunks
    retriever = Retriever()

    retrieved_chunks = retriever.retrieve(
        query=query,
        top_k=5
    )

    # Step 2: Rerank retrieved chunks
    reranker = Reranker()

    reranked_chunks = reranker.rerank(
        query=query,
        retrieved_chunks=retrieved_chunks,
        top_k=3
    )

    # Step 3: Display results
    print("\nReranked Results:\n")

    for chunk in reranked_chunks:

        print("ID:", chunk["id"])
        print("Text:", chunk["text"])
        print("Original Distance:", chunk["distance"])
        print("Rerank Score:", chunk["rerank_score"])

        print("-" * 60)
        