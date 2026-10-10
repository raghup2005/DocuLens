from app.vectorstore import VectorStore


if __name__ == "__main__":

    vector_store = VectorStore()

    test_ids = [
        "test_chunk_1",
        "test_chunk_2",
        "test_chunk_3"
    ]

    vector_store.delete_documents(
        test_ids
    )

    print(
        "Remaining documents:",
        vector_store.count()
    )