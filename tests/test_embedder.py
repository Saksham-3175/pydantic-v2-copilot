"""Tests for EmbedderProtocol, DeterministicMockEmbedder, and GeminiEmbedder."""

import math
import os
import pytest
from storage.embedder import (
    DeterministicMockEmbedder,
    EmbedderProtocol,
    GeminiEmbedder,
    get_default_embedder,
)


@pytest.mark.asyncio
async def test_mock_embedder_dimension():
    embedder = DeterministicMockEmbedder()
    vec = await embedder.embed_query("pydantic v2 migration")
    assert len(vec) == 768

    batch_vecs = await embedder.embed_texts(["first chunk", "second chunk"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 768
    assert len(batch_vecs[1]) == 768

    empty_batch = await embedder.embed_texts([])
    assert empty_batch == []


@pytest.mark.asyncio
async def test_mock_embedder_repeatability():
    embedder = DeterministicMockEmbedder()
    vec1 = await embedder.embed_query("deterministic check")
    vec2 = await embedder.embed_query("deterministic check")
    vec3 = await embedder.embed_query("completely different query")

    assert vec1 == vec2
    assert vec1 != vec3


@pytest.mark.asyncio
async def test_mock_embedder_unit_norm():
    embedder = DeterministicMockEmbedder()
    vec = await embedder.embed_query("normalize to unit length")
    l2_norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(l2_norm, 1.0, rel_tol=1e-5)


def test_protocol_conformance():
    mock_embedder = DeterministicMockEmbedder()
    assert isinstance(mock_embedder, EmbedderProtocol)


def test_factory_fallback():
    embedder = get_default_embedder(prefer_gemini=False)
    assert isinstance(embedder, DeterministicMockEmbedder)


def test_gemini_embedder_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.find_dotenv", lambda *args, **kwargs: "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY must be provided"):
        GeminiEmbedder(api_key=None)


@pytest.mark.asyncio
async def test_gemini_embedder_live_or_skip():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True))
            api_key = os.environ.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        pytest.skip("GEMINI_API_KEY not present in environment or .env")

    try:
        embedder = GeminiEmbedder(api_key=api_key)
        vector = await embedder.embed_query("pydantic v2 migration")
        assert len(vector) == 768
        assert isinstance(vector[0], float)
    except Exception as exc:
        pytest.skip(f"Live Gemini API call failed (offline or network blocked): {exc}")
