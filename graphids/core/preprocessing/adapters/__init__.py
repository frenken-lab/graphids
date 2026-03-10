"""Domain adapters for preprocessing raw data into the IR format."""

from .base import DomainAdapter
from .can_bus import CANBusAdapter

__all__ = ["DomainAdapter", "CANBusAdapter"]
