"""
backend/knowledge_base/vector_store.py

Local vector-store backend for the Sovereign AI Workbench.

Architecture:
    Document chunks
        -> BGE-M3 embeddings
        -> ChromaDB
        -> semantic similarity search
        -> Retriever

This module implements the VectorStoreBackend contract expected by
backend/knowledge_base/retriever.py.

Design requirements:
    - Fully local / offline at runtime.
    - ChromaDB is used as the persistent vector store.
    - BGE-M3 is used as the local multilingual embedding model.
    - No cloud APIs or external network calls.
    - Query results are normalized to the exact shape expected by Retriever:
        {
            "document_id": str,
            "title": str,
            "snippet": str,
            "score": float
        }

Important:
    This module stores vectors and performs search.
    Document parsing/chunking/ingestion belongs in ingestion.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

__all__ = ["VectorStore", "get_vector_store"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Persistent local ChromaDB directory.
VECTOR_DB_DIR = Path(
    os.getenv(
        "KB_VECTOR_DB_DIR",
        str(PROJECT_ROOT / "data" / "knowledge_base"),
    )
)

# Chroma collection name.
COLLECTION_NAME = os.getenv(
    "KB_COLLECTION_NAME",
    "workbench_knowledge",
)

# Local BGE-M3 model path.
DEFAULT_EMBEDDING_MODEL = (
    PROJECT_ROOT / "models" / "embedding" / "bge-m3"
)

EMBEDDING_MODEL_PATH = os.getenv(
    "KB_EMBEDDING_MODEL",
    str(DEFAULT_EMBEDDING_MODEL),
)

# Blueprint specifies BGE-M3 1024-dimensional embeddings.
EMBEDDING_DIMENSION = 1024

DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------


class VectorStore:
    """
    Local ChromaDB vector store using BGE-M3 embeddings.

    This class implements the interface required by Retriever:

        search(query, top_k, file_id=None)
        is_ready()

    Additional write functionality is exposed through add_documents(),
    which ingestion.py can use later.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model_path: str | Path | None = None,
    ) -> None:

        self.persist_directory = Path(
            persist_directory or VECTOR_DB_DIR
        ).resolve()

        self.collection_name = collection_name

        self.embedding_model_path = Path(
            embedding_model_path or EMBEDDING_MODEL_PATH
        ).resolve()

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Initializing ChromaDB at %s",
            self.persist_directory,
        )

        # Persistent local ChromaDB client.
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": (
                    "Sovereign AI Workbench local knowledge base"
                ),
                "embedding_model": "BGE-M3",
                "embedding_dimension": EMBEDDING_DIMENSION,
                "distance": "cosine",
            },
        )

        self._embedding_model: Optional[SentenceTransformer] = None

        logger.info(
            "Vector store initialized. Collection=%s",
            self.collection_name,
        )

    # -----------------------------------------------------------------------
    # Embedding model
    # -----------------------------------------------------------------------

    def _get_embedding_model(self) -> SentenceTransformer:
        """
        Lazily load the local BGE-M3 embedding model.

        Loading is delayed until an embedding is actually required so that
        importing the backend does not immediately consume GPU/RAM.
        """

        if self._embedding_model is not None:
            return self._embedding_model

        if not self.embedding_model_path.exists():
            raise FileNotFoundError(
                "Local BGE-M3 embedding model was not found at: "
                f"{self.embedding_model_path}"
            )

        logger.info(
            "Loading local BGE-M3 embedding model from %s",
            self.embedding_model_path,
        )

        self._embedding_model = SentenceTransformer(
            str(self.embedding_model_path)
        )

        logger.info(
            "BGE-M3 embedding model loaded successfully."
        )

        return self._embedding_model

    # -----------------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------------

    def _embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Convert document chunks into normalized BGE-M3 vectors.
        """

        if not texts:
            return []

        model = self._get_embedding_model()

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def _embed_query(self, query: str) -> List[float]:
        """
        Convert a search query into a normalized BGE-M3 vector.
        """

        if not query.strip():
            return []

        model = self._get_embedding_model()

        embedding = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    # -----------------------------------------------------------------------
    # Write / indexing API
    # -----------------------------------------------------------------------

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Add document chunks to ChromaDB.

        Expected document shape:

        {
            "id": "unique-chunk-id",
            "document_id": "source-document-id",
            "title": "Document title",
            "text": "Chunk text",
            "file_id": "uploaded-file-id",       # optional
            "page": 1,                           # optional
            "chunk_index": 0                     # optional
        }

        Returns:
            Number of chunks indexed.
        """

        if not documents:
            return 0

        texts: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for index, document in enumerate(documents):

            text = str(document.get("text", "")).strip()

            if not text:
                logger.warning(
                    "Skipping empty document chunk at index %d.",
                    index,
                )
                continue

            document_id = str(
                document.get(
                    "document_id",
                    document.get("id", f"document-{index}"),
                )
            )

            chunk_id = str(
                document.get(
                    "id",
                    f"{document_id}-chunk-{index}",
                )
            )

            title = str(
                document.get(
                    "title",
                    "Untitled",
                )
            )

            metadata: Dict[str, Any] = {
                "document_id": document_id,
                "title": title,
            }

            # Optional metadata.
            for key in (
                "file_id",
                "page",
                "chunk_index",
                "source_path",
                "category",
            ):
                value = document.get(key)

                if value is not None:
                    metadata[key] = str(value)

            texts.append(text)
            ids.append(chunk_id)
            metadatas.append(metadata)

        if not texts:
            return 0

        embeddings = self._embed_documents(texts)

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Indexed %d document chunks into ChromaDB.",
            len(texts),
        )

        return len(texts)

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int,
        file_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search required by Retriever.

        Returns:
            List of dictionaries containing:

                document_id
                title
                snippet
                score
        """

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            return []

        if self._collection.count() == 0:
            logger.debug(
                "Knowledge base collection is empty."
            )
            return []

        query_embedding = self._embed_query(query)

        if not query_embedding:
            return []

        where: Optional[Dict[str, Any]] = None

        if file_id:
            where = {
                "file_id": str(file_id),
            }

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        results: List[Dict[str, Any]] = []

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            metadata = metadata or {}

            document_id = str(
                metadata.get(
                    "document_id",
                    "unknown",
                )
            )

            title = str(
                metadata.get(
                    "title",
                    "Untitled",
                )
            )

            snippet = str(text or "")[:400]

            # Chroma cosine distance:
            # lower distance = more similar.
            #
            # Convert to a simple similarity-style score:
            # similarity = 1 - cosine_distance
            score = max(
                0.0,
                min(
                    1.0,
                    1.0 - float(distance),
                ),
            )

            results.append(
                {
                    "document_id": document_id,
                    "title": title,
                    "snippet": snippet,
                    "score": round(score, 4),
                }
            )

        return results

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    async def is_ready(self) -> bool:
        """
        Return True when ChromaDB is available.

        The embedding model itself is intentionally loaded lazily.
        """

        try:
            self._collection.count()
            return True

        except Exception:
            logger.exception(
                "ChromaDB readiness check failed."
            )
            return False

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    async def count(self) -> int:
        """Return the number of indexed chunks."""

        return int(self._collection.count())

    async def delete_document(
        self,
        document_id: str,
    ) -> None:
        """Delete all chunks belonging to a document."""

        self._collection.delete(
            where={
                "document_id": str(document_id),
            }
        )

        logger.info(
            "Deleted document %s from vector store.",
            document_id,
        )

    async def clear(self) -> None:
        """
        Delete and recreate the collection.

        Intended for development/testing only.
        """

        self._client.delete_collection(
            self.collection_name
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": (
                    "Sovereign AI Workbench local knowledge base"
                ),
                "embedding_model": "BGE-M3",
                "embedding_dimension": EMBEDDING_DIMENSION,
                "distance": "cosine",
            },
        )

        logger.warning(
            "Knowledge base vector collection cleared."
        )


# ---------------------------------------------------------------------------
# Singleton / dependency factory
# ---------------------------------------------------------------------------

_default_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Return the process-wide VectorStore singleton.

    Retriever imports this function automatically and uses the returned
    object when it satisfies the VectorStoreBackend protocol.
    """

    global _default_vector_store

    if _default_vector_store is None:
        _default_vector_store = VectorStore()

    return _default_vector_store