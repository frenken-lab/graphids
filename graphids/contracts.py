"""Versioned contract envelopes for boundary specs."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

SpecT = TypeVar("SpecT", bound=BaseModel)


class ContractEnvelope(BaseModel):
    """Versioned wrapper for serialized contract payloads."""

    model_config = ConfigDict(extra="forbid")

    contract: str
    version: int
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


def to_envelope(spec: BaseModel, *, metadata: dict[str, Any] | None = None) -> ContractEnvelope:
    """Wrap a spec in a versioned contract envelope."""
    contract = getattr(spec, "CONTRACT_NAME", None)
    version = getattr(spec, "CONTRACT_VERSION", None)
    if not contract or version is None:
        raise ValueError(f"Spec {type(spec).__name__} missing CONTRACT_NAME/CONTRACT_VERSION")
    return ContractEnvelope(
        contract=contract,
        version=version,
        payload=spec.model_dump(mode="json"),
        metadata=metadata or {},
    )


def from_envelope(payload: dict[str, Any], spec_cls: type[SpecT]) -> SpecT:
    """Validate and deserialize a contract envelope into a spec."""
    envelope = ContractEnvelope.model_validate(payload)
    contract = getattr(spec_cls, "CONTRACT_NAME", None)
    version = getattr(spec_cls, "CONTRACT_VERSION", None)
    if envelope.contract != contract:
        raise ValueError(f"Unexpected contract {envelope.contract!r}; expected {contract!r}")
    if envelope.version != version:
        raise ValueError(f"Unsupported contract version {envelope.version}; expected {version}")
    return spec_cls.model_validate(envelope.payload)
