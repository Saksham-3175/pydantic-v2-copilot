"""Text embedding engines and protocols for Pydantic v2 Copilot."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import typing

from google import genai
from google.genai import types


@typing.runtime_checkable
class EmbedderProtocol(typing.Protocol):
    """Structural interface (socket) for text embedding engines."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into 768-dimensional float vectors."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a 768-dimensional float vector."""
        ...


class DeterministicMockEmbedder:
    """Deterministic, 100% offline mock embedder generating 768-dimensional float vectors.

    Produces pseudo-random float vectors in [-1.0, 1.0] derived from SHA-256 hash
    of input text and normalizes them to unit L2 length.
    """

    DIMENSION: int = 768

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def _text_to_vector(self, text: str) -> list[float]:
        """Convert string to normalized pseudo-random float vector."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return [0.0] * self.dimensions
        return [x / norm for x in vec]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into 768-dimensional unit float vectors."""
        return [self._text_to_vector(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a 768-dimensional unit float vector."""
        return self._text_to_vector(query)


class GeminiEmbedder:
    """Production embedder wrapping the official Google GenAI SDK (gemini-embedding-001)."""

    def __init__(
        self,
        api_key: typing.Optional[str] = None,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            try:
                from dotenv import find_dotenv, load_dotenv

                load_dotenv(find_dotenv(usecwd=True))
                key = os.environ.get("GEMINI_API_KEY")
            except Exception:
                pass

        if not key:
            raise ValueError(
                "GEMINI_API_KEY must be provided or set in environment / .env"
            )

        self.api_key = key
        # Handle deprecated text-embedding-004 by aliasing to gemini-embedding-001
        if model in ("text-embedding-004", "gemini-embedding-001", "gemini-embedding-2"):
            self.model = "gemini-embedding-001" if model == "text-embedding-004" else model
        else:
            self.model = model
        self.dimensions = dimensions
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _normalize_l2(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into 768-dimensional float vectors."""
        if not texts:
            return []
        config = types.EmbedContentConfig(output_dimensionality=self.dimensions)
        response = await self.client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=config,
        )
        if hasattr(response, "embeddings") and response.embeddings:
            return [self._normalize_l2(e.values) for e in response.embeddings]
        if hasattr(response, "embedding") and response.embedding:
            return [self._normalize_l2(response.embedding.values)]
        return []

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a 768-dimensional float vector."""
        config = types.EmbedContentConfig(output_dimensionality=self.dimensions)
        response = await self.client.aio.models.embed_content(
            model=self.model,
            contents=query,
            config=config,
        )
        if hasattr(response, "embeddings") and response.embeddings:
            return self._normalize_l2(response.embeddings[0].values)
        if hasattr(response, "embedding") and response.embedding:
            return self._normalize_l2(response.embedding.values)
        raise ValueError("No embeddings returned by Gemini API")


class CachedEmbedder:
    """Embedder that retrieves precomputed real embeddings from a JSON cache file for 100% offline testing."""

    def __init__(
        self,
        cache_path: typing.Union[str, Path],
        fallback_embedder: typing.Optional[EmbedderProtocol] = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        self._query_cache: dict[str, list[float]] = {}
        query_cache_path = self.cache_path.parent / "gemini_query_cache.json"
        if query_cache_path.exists():
            with open(query_cache_path, "r", encoding="utf-8") as f:
                self._query_cache = json.load(f)
        self._fallback: EmbedderProtocol = fallback_embedder or DeterministicMockEmbedder()
        self._text_to_chunk_id: dict[str, str] = {}
        self._init_corpus_mapping()

    def _init_corpus_mapping(self) -> None:
        try:
            data_dir = self.cache_path.parent
            if data_dir.exists() and any(data_dir.glob("*.md")):
                from storage.chunker import MarkdownHeaderChunker

                chunks = MarkdownHeaderChunker().chunk_directory(data_dir)
                for c in chunks:
                    self._text_to_chunk_id[c.search_text] = c.chunk_id
                    self._text_to_chunk_id[c.chunk_id] = c.chunk_id
        except Exception:
            pass

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts using cache when available, falling back gracefully to mock embedder."""
        results: list[list[float]] = []
        for t in texts:
            if t in self._cache:
                results.append(self._cache[t])
            elif t in self._text_to_chunk_id and self._text_to_chunk_id[t] in self._cache:
                results.append(self._cache[self._text_to_chunk_id[t]])
            elif hasattr(self._fallback, "_text_to_vector"):
                results.append(self._fallback._text_to_vector(t))
            else:
                vecs = await self._fallback.embed_texts([t])
                results.append(vecs[0])
        return results

    async def embed_query(self, query: str) -> list[float]:
        """Embed single query string using cache, live Gemini if API key present, or mock fallback."""
        if query in self._cache:
            return self._cache[query]
        if query in self._query_cache:
            return self._query_cache[query]
        if query in self._text_to_chunk_id and self._text_to_chunk_id[query] in self._cache:
            return self._cache[self._text_to_chunk_id[query]]

        # If custom fallback embedder provided (and not default DeterministicMockEmbedder), use it
        if not isinstance(self._fallback, DeterministicMockEmbedder):
            return await self._fallback.embed_query(query)

        # If real API key is present in environment or .env, attempt live Gemini query embedding
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            try:
                from dotenv import find_dotenv, load_dotenv

                load_dotenv(find_dotenv(usecwd=True))
                api_key = os.environ.get("GEMINI_API_KEY")
            except Exception:
                pass

        if api_key:
            try:
                gemini = GeminiEmbedder(api_key=api_key)
                return await gemini.embed_query(query)
            except Exception:
                pass

        q_norm = query.strip().lower()
        for k, v in self._query_cache.items():
            if k.strip().lower() == q_norm or k.strip().lower() in q_norm or q_norm in k.strip().lower():
                return v

        return self._fallback._text_to_vector(query)


def get_default_embedder(prefer_gemini: bool = True) -> EmbedderProtocol:
    """Factory helper returning GeminiEmbedder if api key is available and prefer_gemini=True, else mock."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True))
            api_key = os.environ.get("GEMINI_API_KEY")
        except Exception:
            pass

    if prefer_gemini and api_key:
        try:
            return GeminiEmbedder(api_key=api_key)
        except Exception:
            return DeterministicMockEmbedder()

    return DeterministicMockEmbedder()

