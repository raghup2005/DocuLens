import chromadb

from app.ingestion.embedder import Embedder


class VectorStore:
    """
    Handles storing and searching document embeddings
    using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = "vectorstore",
        collection_name: str = "rag_documents_v2"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.embedder = Embedder()

    # --------------------------------
    # Add documents
    # --------------------------------

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str]
    ):
        """
        Add new documents to ChromaDB.

        IDs must be unique.
        """

        embeddings = self.embedder.embed_documents(
            texts
        )

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    # --------------------------------
    # Upsert documents
    # --------------------------------

    def upsert_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str]
    ):
        """
        Insert new documents or update existing
        documents with the same IDs.
        """

        embeddings = self.embedder.embed_documents(
            texts
        )

        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    # --------------------------------
    # Search
    # --------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        Perform dense semantic search.
        """

        query_embedding = self.embedder.embed_text(
            query
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results

    # --------------------------------
    # Count
    # --------------------------------

    def count(self) -> int:
        """
        Return total number of documents
        stored in the collection.
        """

        return self.collection.count()

    # --------------------------------
    # Delete documents
    # --------------------------------

    def delete_documents(
        self,
        ids: list[str]
    ):
        """
        Delete documents from ChromaDB using IDs.
        """

        self.collection.delete(
            ids=ids
        )

    # --------------------------------
    # Get all documents
    # --------------------------------

    def get_all_documents(self):
        """
        Return all documents, metadata, and IDs
        from the collection.

        Useful for BM25 index creation and debugging.
        """

        return self.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )


# ====================================
# TEST
# ====================================

if __name__ == "__main__":

    vector_store = VectorStore()

    print(
        "\nCollection:",
        vector_store.collection.name
    )

    print(
        "Documents stored:",
        vector_store.count()
    )

    # --------------------------------
    # Test semantic search only
    # --------------------------------

    if vector_store.count() > 0:

        results = vector_store.search(
            query="How does self attention work?",
            top_k=2
        )

        print("\nSearch results:\n")

        if results["documents"]:

            for document in results["documents"][0]:

                print(
                    "-",
                    document
                )

        else:

            print(
                "No documents found."
            )

    else:

        print(
            "\nChromaDB is empty."
        )

        print(
            "Run your ingestion pipeline first."
        )