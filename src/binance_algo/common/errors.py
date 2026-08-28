"""Project exception hierarchy."""


class BinanceAlgoError(Exception):
    """Base class for failures that callers are expected to handle."""


class ConfigurationError(BinanceAlgoError):
    """Configuration could not be loaded or violates a safety invariant."""


class DataContractError(BinanceAlgoError):
    """An upstream payload is incompatible with the canonical contract."""


class StorageError(BinanceAlgoError):
    """A durable write or validation step failed."""


class UniverseError(BinanceAlgoError):
    """A point-in-time universe could not be built."""


class StateStoreError(BinanceAlgoError):
    """Operational state could not be read or changed transactionally."""


class InvalidStateTransition(StateStoreError):
    """A manifest or job transition violates its state machine."""


class ArchiveError(BinanceAlgoError):
    """An official public archive could not be downloaded or validated."""


class DataQualityError(BinanceAlgoError):
    """Normalized market data violates its canonical contract or quality gate."""
