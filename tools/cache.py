import time
from typing import Any, Dict, Tuple


class TTLCache:
    """Simple in-memory TTL cache with coarse eviction.

    This is intentionally lightweight and process-local. It avoids adding a
    Redis dependency while still preventing redundant upstream API calls.
    """

    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def _now(self) -> float:
        return time.time()

    def get(self, key: str) -> Any:
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if self._now() - ts > self.ttl:
            # Expired
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (self._now(), value)


# Shared caches for ticker-centric data
ticker_price_cache = TTLCache(ttl_seconds=60)
ticker_info_cache = TTLCache(ttl_seconds=60)
historical_cache = TTLCache(ttl_seconds=60)

