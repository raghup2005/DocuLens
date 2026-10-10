from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Converts text into numerical vector embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def embed_text(
        self,
        text: str
    ) -> list[float]:
        """
        Convert a single text into an embedding.
        """

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Convert multiple text chunks into embeddings.
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()
if __name__ == "__main__":

    embedder = Embedder()

    text = "Transformers use self-attention mechanisms."

    embedding = embedder.embed_text(text)

    print("Embedding dimension:")
    print(len(embedding))

    print("\nFirst 10 values:")
    print(embedding[:10])