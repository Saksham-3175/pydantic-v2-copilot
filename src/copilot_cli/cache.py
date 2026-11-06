"""Content-addressed SHA-256 cache for verified Pydantic v2 migrations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Optional


class MigrationCache:
    """Stores verified migration drafts keyed by file path and content SHA-256."""

    def __init__(self, cache_file: Optional[Path] = None) -> None:
        if cache_file is not None:
            self.cache_file = cache_file
        else:
            base_dir = Path.home() / ".cache" / "pydantic_copilot"
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
                self.cache_file = base_dir / "cache.json"
            except Exception:
                self.cache_file = Path(tempfile.gettempdir()) / "pydantic_copilot_cache.json"

    def _compute_key(self, file_path: Path, content: str) -> str:
        resolved = str(file_path.resolve())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{resolved}\n{content_hash}".encode("utf-8")).hexdigest()

    def get(self, file_path: Path, content: str) -> Optional[dict]:
        """Retrieve cached modernized code if content hash matches."""
        if not self.cache_file.exists():
            return None
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            key = self._compute_key(file_path, content)
            return data.get(key)
        except Exception:
            return None

    def set(self, file_path: Path, content: str, modern_code: str, iterations: int) -> None:
        """Store verified modernized code."""
        try:
            data = {}
            if self.cache_file.exists():
                try:
                    data = json.loads(self.cache_file.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            key = self._compute_key(file_path, content)
            data[key] = {
                "file_path": str(file_path.resolve()),
                "modern_code": modern_code,
                "iterations": iterations,
            }
            # Ensure parent exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
