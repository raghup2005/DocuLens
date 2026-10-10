from app.vectorstore import VectorStore


class Retriever:
    """
    Retrieves relevant document chunks
    from ChromaDB using semantic similarity.
    """

    def __init__(self):

        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
        distance_threshold: float = 1.2,
        where: dict | None = None
    ) -> list:
        """
        Retrieve relevant chunks using semantic similarity
        and distance filtering.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        if distance_threshold < 0:
            raise ValueError(
                "distance_threshold cannot be negative."
            )

        # --------------------------------
        # Query embedding
        # --------------------------------

        query_embedding = (
            self.vector_store.embedder.embed_text(
                query
            )
        )

        # --------------------------------
        # Dense search
        # --------------------------------

        if where:

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                where=where
            )

        else:

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k
            )

        if not results["documents"]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        retrieved_chunks = []

        # --------------------------------
        # Distance filtering
        # --------------------------------

        for i in range(len(documents)):

            distance = float(
                distances[i]
            )

            if distance > distance_threshold:
                continue

            retrieved_chunks.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distance
            })

        # --------------------------------
        # Sort by distance
        # --------------------------------

        retrieved_chunks.sort(
            key=lambda x: x["distance"]
        )

        return retrieved_chunks[:top_k]


if __name__ == "__main__":

    retriever = Retriever()

    query = "What is self attention?"

    results = retriever.retrieve(
        query=query,
        top_k=5,
        candidate_k=10,
        distance_threshold=1.2,
        where={
            "source": "1706.03762v7.pdf"
        }
    )

    print("\nRetrieved Results:\n")

    if not results:

        print(
            "No chunks passed the similarity threshold."
        )

    for result in results:

        print("ID:", result["id"])
        print("Distance:", result["distance"])
        print("Metadata:", result["metadata"])
        print("Text:", result["text"])

        print("-" * 60)
        