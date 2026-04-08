class StateManagerError(Exception):
    """Base error for state manager failures."""


class EvictedContextError(StateManagerError):
    """Raised when a requested context node is not cached."""


class InvalidIgnoreSpanError(StateManagerError):
    """Raised when ignore spans are malformed."""
