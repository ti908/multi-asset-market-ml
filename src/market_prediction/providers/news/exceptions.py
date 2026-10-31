"""Exception hierarchy for market-news providers."""


class NewsProviderError(Exception):
    """Base exception for expected news-provider failures."""


class NewsProviderUnavailableError(NewsProviderError):
    """Raised when a news provider is temporarily unavailable."""


class NewsQuotaExceededError(NewsProviderError):
    """Raised when a provider quota or rate allowance is exhausted."""


class UnsupportedNewsQueryError(NewsProviderError):
    """Raised when a provider cannot support the requested query."""


class InvalidNewsProviderDataError(NewsProviderError):
    """Raised when upstream news data violates the provider contract."""