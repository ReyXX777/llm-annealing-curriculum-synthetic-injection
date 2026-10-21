"""
Registry for managing multiple streaming sources.
"""
from __future__ import annotations
from typing import Dict, Iterator
import logging
from .base import StreamingSource
from ..exceptions import EmptySourceError
from ..types import SourceType

logger = logging.getLogger(__name__)

class SourceRegistry:
    """
    Holds all data sources D. Provides unified sampling interface.
    """
    def __init__(self, sources: Dict[str, StreamingSource]):
        if not sources:
            raise ValueError("sources dict cannot be empty")
        self.sources: Dict[str, StreamingSource] = sources

    def get(self, key: str) -> StreamingSource:
        if key not in self.sources:
            raise KeyError(f"source {key} not in registry, available: {list(self.sources.keys())}")
        return self.sources[key]

    def available_keys(self):
        return list(self.sources.keys())

    def sample_from(self, key: str):
        """Sample one example from key, with graceful handling."""
        try:
            src = self.get(key)
            return src.sample()
        except EmptySourceError as e:
            logger.warning(f"Source {key} empty, skipping: {e}")
            raise
        except Exception as e:
            logger.error(f"Error sampling from {key}: {e}", exc_info=True)
            raise EmptySourceError(str(e)) from e
