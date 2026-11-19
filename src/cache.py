import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class QueryCache:
    """
    Persistent JSON cache for repeated search and answer queries.
    """

    def __init__(self, cache_file: str = "data/processed/query_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads cached entries from disk if the file exists."""
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        """Persists the current cache dictionary to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save query cache: {e}")

    def _make_key(self, namespace: str, query: str, k: int) -> str:
        """Creates a deterministic hash key for (namespace, query, k)."""
        raw = f"{namespace}:{k}:{query.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, namespace: str, query: str, k: int) -> Optional[Any]:
        """Retrieves a cached result if present."""
        key = self._make_key(namespace, query, k)
        return self._cache.get(key)

    def set(self, namespace: str, query: str, k: int, value: Any) -> None:
        """Stores a result in cache and persists it."""
        key = self._make_key(namespace, query, k)
        self._cache[key] = value
        self._save()
