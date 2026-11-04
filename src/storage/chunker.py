"""Hierarchical markdown header chunker with AST-aware code block preservation and negation-trap decoupling."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional


@dataclass
class DocChunk:
    chunk_id: str          # e.g., "pydantic_v2_migration#field-validation"
    topic: str             # e.g., "pydantic_validators", "pydantic_config", "fastapi_lifespan"
    source_file: str       # e.g., "pydantic_v2_migration.md"
    breadcrumb: str        # e.g., "Pydantic V2 Migration Guide > Field Validation > Modern Field Validator (@field_validator)"
    search_text: str       # Clean, modern v2 signatures only (prevents Negation Trap)
    payload_content: str   # Full markdown text including Context breadcrumb and Before/After examples

    def to_dict(self) -> dict[str, str]:
        """Serialize chunk to Qdrant payload dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "topic": self.topic,
            "source_file": self.source_file,
            "breadcrumb": self.breadcrumb,
            "content": self.payload_content,
        }


def slugify(text: str) -> str:
    """Generate a clean URL/anchor slug from a heading string."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def map_topic(current_h2: Optional[str], current_h3: Optional[str] = None, file_name: str = "") -> str:
    """Map section/subtopic headers and filename to normalized topic taxonomy."""
    h2 = (current_h2 or "").lower()
    h3 = (current_h3 or "").lower()
    fn = file_name.lower()
    combined = f"{fn} {h2} {h3}"

    if "root and model" in combined or "model_validator" in combined or "model validator" in combined:
        return "pydantic_model_validators"
    if "field validation" in combined or "field_validator" in combined or "validators" in combined:
        return "pydantic_validators"
    if "settings" in combined or "basesettings" in combined or "environment" in combined or "pydantic_settings" in fn:
        return "pydantic_settings"
    if "configuration" in combined or "configdict" in combined or "config" in combined:
        return "pydantic_config"
    if "serialization" in combined or "model_dump" in combined or "dump" in combined:
        return "pydantic_serialization"
    if "lifespan" in combined:
        return "fastapi_lifespan"
    if "dependency" in combined or "dependencies" in combined or "annotated" in combined:
        return "fastapi_dependencies"

    if current_h2:
        return slugify(current_h2).replace("-", "_")
    return "general"


def derive_chunk_id(
    file_stem: str,
    current_h2: Optional[str],
    current_h3: Optional[str] = None,
    seen_ids: Optional[set[str]] = None,
) -> str:
    """Derive unique anchor slug from section/subtopic headers."""
    if current_h2:
        base_slug = slugify(current_h2)
    elif current_h3:
        base_slug = slugify(current_h3)
    else:
        base_slug = "overview"

    candidate_id = f"{file_stem}#{base_slug}"

    if seen_ids is not None:
        if candidate_id in seen_ids and current_h3:
            sub_slug = slugify(current_h3)
            candidate_id = f"{file_stem}#{base_slug}--{sub_slug}"

        counter = 2
        orig_id = candidate_id
        while candidate_id in seen_ids:
            candidate_id = f"{orig_id}-{counter}"
            counter += 1
        seen_ids.add(candidate_id)

    return candidate_id


def is_legacy_code_block(code_text: str) -> bool:
    """Detect whether a fenced code block represents legacy/deprecated code."""
    lines = code_text.splitlines()
    header_lines = "\n".join(lines[:6]).lower()
    markers = [
        "legacy",
        "deprecated",
        "not recommended",
    ]
    return any(marker in header_lines for marker in markers)


def extract_search_text(breadcrumb: str, raw_text: str) -> str:
    """Build clean search_text for vector embedding.

    Decouples modern code and concepts from legacy patterns to prevent the Negation Trap.
    """
    clean_parts: list[str] = [f"Context: [{breadcrumb}]"]

    lines = raw_text.splitlines()
    i = 0
    prose_accumulator: list[str] = []

    def flush_prose():
        if prose_accumulator:
            text = "\n".join(prose_accumulator).strip()
            if text:
                clean_parts.append(text)
            prose_accumulator.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_prose()
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if lines[i].strip().startswith("```"):
                    break
                i += 1

            code_block_text = "\n".join(code_lines)
            if not is_legacy_code_block(code_block_text):
                clean_parts.append(code_block_text)
            i += 1
            continue

        prose_accumulator.append(line)
        i += 1

    flush_prose()
    return "\n\n".join(part for part in clean_parts if part)


class MarkdownHeaderChunker:
    """State-machine markdown parser that partitions documents by headers while preserving atomic code blocks."""

    def chunk_file(self, file_path: Path | str) -> list[DocChunk]:
        """Parse a markdown file into structured DocChunk objects."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Documentation file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        doc_title: str = ""
        current_h2: Optional[str] = None
        current_h3: Optional[str] = None
        in_code_block: bool = False
        buffer: list[str] = []
        chunks: list[DocChunk] = []
        seen_ids: set[str] = set()

        def flush_chunk():
            nonlocal buffer
            raw_text = "\n".join(buffer).strip()
            if raw_text.endswith("---"):
                raw_text = raw_text[:-3].strip()
            buffer.clear()

            if not raw_text:
                return

            parts = [p for p in [doc_title, current_h2, current_h3] if p]
            breadcrumb = " > ".join(parts) if parts else file_path.stem

            chunk_id = derive_chunk_id(
                file_stem=file_path.stem,
                current_h2=current_h2,
                current_h3=current_h3,
                seen_ids=seen_ids,
            )

            topic = map_topic(
                current_h2=current_h2,
                current_h3=current_h3,
                file_name=file_path.name,
            )

            payload_content = f"Context: [{breadcrumb}]\n\n" + raw_text
            search_text = extract_search_text(breadcrumb=breadcrumb, raw_text=raw_text)

            # Invariant check: Ensure code fences are balanced
            payload_fences = payload_content.count("```")
            assert payload_fences % 2 == 0, (
                f"Unbalanced code fences ({payload_fences}) in payload_content for {chunk_id}"
            )
            search_fences = search_text.count("```")
            assert search_fences % 2 == 0, (
                f"Unbalanced code fences ({search_fences}) in search_text for {chunk_id}"
            )

            chunk = DocChunk(
                chunk_id=chunk_id,
                topic=topic,
                source_file=file_path.name,
                breadcrumb=breadcrumb,
                search_text=search_text,
                payload_content=payload_content,
            )
            chunks.append(chunk)

        for line in lines:
            stripped = line.strip()

            # Code fence toggle
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                buffer.append(line)
                continue

            if in_code_block:
                # Code lines must be kept 100% intact, never treat '#' as header
                buffer.append(line)
                continue

            # Header detection outside of code blocks
            if line.startswith("# "):
                doc_title = line[2:].strip()
                continue
            elif line.startswith("## "):
                flush_chunk()
                current_h2 = line[3:].strip()
                current_h3 = None
                continue
            elif line.startswith("### "):
                flush_chunk()
                current_h3 = line[4:].strip()
                continue

            buffer.append(line)

        # Flush final chunk at EOF
        flush_chunk()

        return chunks

    def chunk_directory(self, data_dir: Path | str) -> list[DocChunk]:
        """Parse all markdown files in data_dir in deterministic alphabetical order."""
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Data directory not found: {data_dir}")

        all_chunks: list[DocChunk] = []
        for md_file in sorted(data_dir.glob("*.md")):
            file_chunks = self.chunk_file(md_file)
            all_chunks.extend(file_chunks)

        return all_chunks
