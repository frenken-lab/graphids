"""Domain adapters for preprocessing raw data into the IR format."""

from ._can_bus import CANBusAdapter
from .base import DomainAdapter

__all__ = ["DomainAdapter", "CANBusAdapter"]
