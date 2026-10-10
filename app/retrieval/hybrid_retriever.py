import re

from rank_bm25 import BM25Okapi

from app.vectorstore import VectorStore


class HybridRetriever:
    """
    Multimodal hybrid retriever combining:

    1. Dense semantic retrieval using ChromaDB
    2. Sparse keyword retrieval using BM25
    3. Reciprocal Rank Fusion (RRF)
    4. Weighted score fusion
    5. Metadata filtering
    6. Content-type filtering

    Supported content types:

        text
        table
        image
    """

    def __init__(
        self,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):

        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError(
                "Weights cannot be negative."
            )

        if dense_weight + bm25_weight == 0:
            raise ValueError(
                "At least one weight must be greater than 0."
            )

        self.vector_store = VectorStore()

        self.rrf_k = rrf_k

        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.documents = []
        self.metadatas = []
        self.ids = []

        self.bm25 = None

        self._build_bm25_index()

    # --------------------------------
    # Tokenization
    # --------------------------------

    def tokenize(
        self,
        text: str
    ) -> list[str]:
        """
        Tokenize text for BM25.
        """

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    # --------------------------------
    # Metadata filtering
    # --------------------------------

    def _matches_filter(
        self,
        metadata: dict,
        where: dict | None
    ) -> bool:
        """
        Check whether metadata matches a filter.

        Supports:

            {"source": "..."}
            {"page": 5}
            {"content_type": "text"}

        AND filters:

            {
                "$and": [
                    {"source": "..."},
                    {"content_type": "text"}
                ]
            }
        """

        if not where:
            return True

        # --------------------------------
        # Simple equality filters
        # --------------------------------

        if "$and" not in where:

            for key, expected_value in where.items():

                if metadata.get(key) != expected_value:
                    return False

            return True

        # --------------------------------
        # AND filters
        # --------------------------------

        conditions = where["$and"]

        for condition in conditions:

            if not self._matches_filter(
                metadata,
                condition
            ):
                return False

        return True

    # --------------------------------
    # Build metadata filter
    # --------------------------------

   

    def _build_where(
    self,
    source: str | None = None,
    content_type: str | None = None,
    page: int | None = None,
    figure_number: int | None = None
) -> dict | None:
        """
        Build a metadata filter.

        Examples:

            content_type="image"

            source="1706.03762v7.pdf"

            page=5

        Multiple filters use AND logic.
        """

        conditions = []

        if source:

            conditions.append({
                "source": source
            })

        if content_type:

            if content_type not in {
                "text",
                "table",
                "image"
            }:

                raise ValueError(
                    "content_type must be "
                    "'text', 'table', or 'image'."
                )

            conditions.append({
                "content_type": content_type
            })

        if page is not None:

            conditions.append({
                "page": page
            })

        if figure_number is not None:
            conditions.append({"figure_number":figure_number})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {
            "$and": conditions
        }

    # --------------------------------
    # Build BM25 index
    # --------------------------------

    def _build_bm25_index(self):

        data = self.vector_store.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        self.documents = data.get(
            "documents",
            []
        )

        self.metadatas = data.get(
            "metadatas",
            []
        )

        self.ids = data.get(
            "ids",
            []
        )

        if not self.documents:

            self.bm25 = None

            return

        tokenized_documents = [
            self.tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    # --------------------------------
    # Dense retrieval
    # --------------------------------

    def _dense_search(
        self,
        query: str,
        top_k: int,
        where: dict | None = None
    ) -> list:
        """
        Perform dense semantic retrieval.
        """

        query_embedding = (
            self.vector_store.embedder.embed_text(
                query
            )
        )

        # --------------------------------
        # ChromaDB search
        # --------------------------------

        if where:

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )

        else:

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

        if not results["documents"]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        dense_results = []

        for rank, (
            document,
            metadata,
            distance,
            doc_id
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
                ids
            ),
            start=1
        ):

            dense_results.append({
                "id": doc_id,
                "text": document,
                "metadata": metadata,
                "distance": float(distance),
                "dense_rank": rank
            })

        return dense_results

    # --------------------------------
    # BM25 retrieval
    # --------------------------------

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        where: dict | None = None
    ) -> list:
        """
        Perform BM25 retrieval with metadata filtering.
        """

        if self.bm25 is None:
            return []

        query_tokens = self.tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        sparse_results = []

        for index in ranked_indices:

            metadata = self.metadatas[index]

            # --------------------------------
            # Apply metadata filter
            # --------------------------------

            if not self._matches_filter(
                metadata,
                where
            ):
                continue

            rank = len(
                sparse_results
            ) + 1

            sparse_results.append({
                "id": self.ids[index],
                "text": self.documents[index],
                "metadata": metadata,
                "bm25_score": float(
                    scores[index]
                ),
                "bm25_rank": rank
            })

            if len(sparse_results) >= top_k:
                break

        return sparse_results

    # --------------------------------
    # RRF score
    # --------------------------------

    def _rrf_score(
        self,
        rank: int
    ) -> float:

        return 1.0 / (
            self.rrf_k + rank
        )

    # --------------------------------
    # Hybrid retrieval
    # --------------------------------

    
    def retrieve(
    self,
    query: str,
    top_k: int = 5,
    dense_k: int = 10,
    bm25_k: int = 10,
    source: str | None = None,
    content_type: str | None = None,
    page: int | None = None,
    figure_number: int | None = None
) -> list:
        """
        Perform multimodal weighted hybrid retrieval.

        Args:

            query:
                User question.

            top_k:
                Number of final candidates.

            dense_k:
                Number of dense candidates.

            bm25_k:
                Number of BM25 candidates.

            source:
                Optional source filter.

            content_type:
                Optional:
                    text
                    table
                    image

            page:
                Optional page filter.
        """

        if not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        # --------------------------------
        # Build metadata filter
        # --------------------------------

        where = self._build_where(
    source=source,
    content_type=content_type,
    page=page,
    figure_number=figure_number
)

        # --------------------------------
        # Dense search
        # --------------------------------

        dense_results = self._dense_search(
            query=query,
            top_k=dense_k,
            where=where
        )

        # --------------------------------
        # BM25 search
        # --------------------------------

        bm25_results = self._bm25_search(
            query=query,
            top_k=bm25_k,
            where=where
        )

        # --------------------------------
        # Fusion
        # --------------------------------

        fused = {}

        # --------------------------------
        # Dense contribution
        # --------------------------------

        for result in dense_results:

            doc_id = result["id"]

            if doc_id not in fused:

                fused[doc_id] = {
                    "id": doc_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rrf_score": 0.0,
                    "bm25_rrf_score": 0.0,
                    "distance": result["distance"],
                    "bm25_score": 0.0
                }

            dense_rrf = self._rrf_score(
                result["dense_rank"]
            )

            fused[doc_id]["dense_rrf_score"] = (
                dense_rrf
            )

            fused[doc_id]["rrf_score"] += (
                self.dense_weight *
                dense_rrf
            )

        # --------------------------------
        # BM25 contribution
        # --------------------------------

        for result in bm25_results:

            doc_id = result["id"]

            if doc_id not in fused:

                fused[doc_id] = {
                    "id": doc_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rrf_score": 0.0,
                    "bm25_rrf_score": 0.0,
                    "distance": None,
                    "bm25_score": result["bm25_score"]
                }

            bm25_rrf = self._rrf_score(
                result["bm25_rank"]
            )

            fused[doc_id]["bm25_rrf_score"] = (
                bm25_rrf
            )

            fused[doc_id]["bm25_score"] = (
                result["bm25_score"]
            )

            fused[doc_id]["rrf_score"] += (
                self.bm25_weight *
                bm25_rrf
            )

        # --------------------------------
        # Sort
        # --------------------------------

        hybrid_results = list(
            fused.values()
        )

        hybrid_results.sort(
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return hybrid_results[:top_k]


# ====================================
# TEST
# ====================================

if __name__ == "__main__":

    retriever = HybridRetriever(
        dense_weight=0.7,
        bm25_weight=0.3
    )

    # --------------------------------
    # Test 1: All modalities
    # --------------------------------

    print("\n" + "=" * 60)
    print("TEST 1: ALL MODALITIES")
    print("=" * 60)

    results = retriever.retrieve(
        query="What does Figure 2 show?",
        top_k=5,
        dense_k=10,
        bm25_k=10,
        source="1706.03762v7.pdf"
    )

    for result in results:

        print(
            "\nID:",
            result["id"]
        )

        print(
            "Content Type:",
            result["metadata"].get(
                "content_type"
            )
        )

        print(
            "Page:",
            result["metadata"].get(
                "page"
            )
        )

        print(
            "RRF Score:",
            result["rrf_score"]
        )

        print(
            "Text:",
            result["text"][:300]
        )

    # --------------------------------
    # Test 2: Image only
    # --------------------------------

    print("\n" + "=" * 60)
    print("TEST 2: IMAGE ONLY")
    print("=" * 60)

    image_results = retriever.retrieve(
        query="What does Figure 2 show?",
        top_k=5,
        dense_k=10,
        bm25_k=10,
        source="1706.03762v7.pdf",
        content_type="image"
    )

    for result in image_results:

        print(
            "\nID:",
            result["id"]
        )

        print(
            "Content Type:",
            result["metadata"].get(
                "content_type"
            )
        )

        print(
            "Page:",
            result["metadata"].get(
                "page"
            )
        )

        print(
            "Text:",
            result["text"][:500]
        )

        print("-" * 60)

if __name__ == "__main__":

    retriever = HybridRetriever(
        dense_weight=0.7,
        bm25_weight=0.3
    )

    results = retriever.retrieve(
        query="What does Figure 2 show?",
        top_k=5,
        dense_k=10,
        bm25_k=10,
        source="1706.03762v7.pdf",
        content_type="image",
        figure_number=2
    )

    print(
        "\nFigure 2 Retrieval Results:\n"
    )

    for result in results:

        print(
            "ID:",
            result["id"]
        )

        print(
            "Figure:",
            result["metadata"].get(
                "figure_number"
            )
        )

        print(
            "Page:",
            result["metadata"].get(
                "page"
            )
        )

        print(
            "RRF Score:",
            result["rrf_score"]
        )

        print(
            "Text:",
            result["text"][:500]
        )

        print("-" * 60)