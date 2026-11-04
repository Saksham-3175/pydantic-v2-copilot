"""Tests for hierarchical MarkdownHeaderChunker and DocChunk data contract."""

from pathlib import Path
import pytest
from storage.chunker import (
    DocChunk,
    MarkdownHeaderChunker,
    derive_chunk_id,
    extract_search_text,
    map_topic,
    slugify,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "src" / "storage" / "data"


def test_doc_chunk_contract():
    chunk = DocChunk(
        chunk_id="test_doc#field-validation",
        topic="pydantic_validators",
        source_file="test_doc.md",
        breadcrumb="Test Guide > Field Validation > Subtopic",
        search_text="Clean search text",
        payload_content="Context: [Test Guide > Field Validation > Subtopic]\n\nFull content",
    )
    assert chunk.chunk_id == "test_doc#field-validation"
    assert chunk.topic == "pydantic_validators"
    payload = chunk.to_dict()
    assert payload["chunk_id"] == chunk.chunk_id
    assert payload["topic"] == chunk.topic
    assert payload["content"] == chunk.payload_content
    assert payload["source_file"] == chunk.source_file
    assert payload["breadcrumb"] == chunk.breadcrumb


def test_slugify():
    assert slugify("Field Validation") == "field-validation"
    assert slugify("Modern Field Validator (@field_validator)") == "modern-field-validator-field_validator"
    assert slugify("Environment Parsing & Precedence") == "environment-parsing-precedence"
    assert slugify("  Extra   Spaces  ") == "extra-spaces"


def test_map_topic():
    assert map_topic("Field Validation") == "pydantic_validators"
    assert map_topic("Model Configuration") == "pydantic_config"
    assert map_topic("Root and Model Validators") == "pydantic_model_validators"
    assert map_topic("Serialization") == "pydantic_serialization"
    assert map_topic("BaseSettings Migration", file_name="pydantic_settings.md") == "pydantic_settings"
    assert map_topic("Environment Parsing & Precedence", file_name="pydantic_settings.md") == "pydantic_settings"
    assert map_topic("Application Lifespan") == "fastapi_lifespan"
    assert map_topic("Dependency Injection") == "fastapi_dependencies"


def test_derive_chunk_id():
    seen: set[str] = set()
    id1 = derive_chunk_id("file", "Field Validation", seen_ids=seen)
    assert id1 == "file#field-validation"
    id2 = derive_chunk_id("file", "Field Validation", current_h3="Wrap Validator", seen_ids=seen)
    assert id2 == "file#field-validation--wrap-validator"


def test_legacy_filtering_in_search_text():
    breadcrumb = "Guide > Section > Subtopic"
    raw_text = """Some modern overview text.

```python
# Legacy Pydantic v1 Pattern (Deprecated)
from pydantic import BaseModel, validator
class Legacy(BaseModel):
    pass
```

```python
# Modern Pydantic v2 Pattern (Standard)
from pydantic import BaseModel, field_validator
class Modern(BaseModel):
    pass
```
"""
    search_text = extract_search_text(breadcrumb, raw_text)
    assert "Legacy Pydantic v1 Pattern" not in search_text
    assert "class Legacy" not in search_text
    assert "from pydantic import BaseModel, validator" not in search_text
    assert "Modern Pydantic v2 Pattern" in search_text
    assert "field_validator" in search_text
    assert search_text.count("```") == 2  # Balanced code fences


def test_code_fence_preservation(tmp_path: Path):
    doc_path = tmp_path / "code_test.md"
    doc_path.write_text(
        """# Doc Title
## Section 1
Here is code with hashes:
```python
# This is a comment, not an H1 header!
## This is also a comment, not an H2!
x = 1
```
Trailing prose.
""",
        encoding="utf-8",
    )
    chunker = MarkdownHeaderChunker()
    chunks = chunker.chunk_file(doc_path)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.breadcrumb == "Doc Title > Section 1"
    assert "# This is a comment" in chunk.payload_content
    assert "## This is also a comment" in chunk.payload_content
    assert chunk.payload_content.count("```") == 2
    assert chunk.search_text.count("```") == 2


def test_curated_corpus_chunks():
    chunker = MarkdownHeaderChunker()
    chunks = chunker.chunk_directory(CORPUS_DIR)

    assert len(chunks) == 8, f"Expected 8 chunks, got {len(chunks)}"

    expected_ids = [
        "fastapi_modern#application-lifespan",
        "fastapi_modern#dependency-injection",
        "pydantic_settings#basesettings-migration",
        "pydantic_settings#environment-parsing-precedence",
        "pydantic_v2_migration#field-validation",
        "pydantic_v2_migration#model-configuration",
        "pydantic_v2_migration#root-and-model-validators",
        "pydantic_v2_migration#serialization",
    ]

    actual_ids = [c.chunk_id for c in chunks]
    assert actual_ids == expected_ids

    for c in chunks:
        # Balanced fences
        assert c.payload_content.count("```") % 2 == 0
        assert c.search_text.count("```") % 2 == 0
        assert c.payload_content.startswith(f"Context: [{c.breadcrumb}]")
        assert c.search_text.startswith(f"Context: [{c.breadcrumb}]")
        # Ensure search_text filtered out legacy code
        assert "# legacy" not in c.search_text.lower()
        assert "# deprecated" not in c.search_text.lower()
        # Verify non-empty
        assert len(c.search_text) > 100
        assert len(c.payload_content) > 100
