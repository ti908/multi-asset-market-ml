"""Provider-specific exception hierarchy.

Concrete market-data providers must translate implementation-specific
exceptions into these project-level exceptions. Downstream code should
never need to understand exceptions raised by yfinance, HTTP clients,
or any other provider-specific dependency.
"""


class ProviderError(Exception):
    """Base exception for all market-data provider failures."""


class ProviderUnavailableError(ProviderError):
    """Raised when the upstream provider cannot be reached.

    Examples include network failures, connection errors, and request
    timeouts. Retry behavior will be implemented later by the ingestion
    layer rather than by this abstraction.
    """


class UnsupportedAssetError(ProviderError):
    """Raised when a provider cannot serve the requested asset.

    This includes invalid vendor symbols and assets that the provider
    does not support.
    """


class EmptyDataError(ProviderError):
    """Raised when a provider successfully responds but returns no data.

    An empty DataFrame is therefore not considered a successful provider
    response.
    """


class InvalidProviderDataError(ProviderError):
    """Raised when a provider cannot satisfy the provider data contract.

    This protects the abstraction from returning ambiguous results such
    as missing required columns, null required values, duplicate
    timestamps, or only a partial subset of the requested date range
    without explicitly reporting the gap.
    """