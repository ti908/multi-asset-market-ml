"""Validated market-data cache contracts and persistence."""

from market_prediction.cache.manager import CacheManager
from market_prediction.cache.models import (
    CacheMetadata,
    CacheReadResult,
    CacheState,
    ColumnPriceBasis,
    HistoricalStability,
    PriceAdjustment,
    RestatementSummary,
    SoftAnomalySummary,
    ValidatedDataset,
    ValidationGateError,
    YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS,
)

__all__ = [
    "CacheManager",
    "CacheMetadata",
    "CacheReadResult",
    "CacheState",
    "ColumnPriceBasis",
    "HistoricalStability",
    "PriceAdjustment",
    "RestatementSummary",
    "SoftAnomalySummary",
    "ValidatedDataset",
    "ValidationGateError",
    "YFINANCE_AUTO_ADJUST_FALSE_PRICE_BASIS",
]