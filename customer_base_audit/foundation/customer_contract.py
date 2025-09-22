"""Customer contract definitions and validation utilities.

The customer contract captures the minimum pieces of information
that every downstream analytics workflow relies on. It ensures we can
answer the basic audit question "How many customers do we have?"
and "When did a customer first engage with us?" consistently across
source systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class CustomerIdentifier:
    """Canonical representation of a customer record.

    Attributes
    ----------
    customer_id:
        Unique identifier after any cross-system merge.
    acquisition_ts:
        Timestamp of the customer's first observed transaction or
        explicit acquisition event.
    source_system:
        Originating system for lineage/auditability purposes.
    is_visible:
        Indicates whether the customer is currently visible in the
        operational systems (e.g., opted in, not anonymised).
    metadata:
        Optional metadata that downstream processes may leverage
        (e.g., loyalty tier, region). The contract does not prescribe
        the shape of metadata so that teams can extend it when needed.
    """

    customer_id: str
    acquisition_ts: datetime
    source_system: str
    is_visible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalised_metadata(self) -> Mapping[str, Any]:
        """Return a shallow copy of metadata with string keys.

        Ensures accidental use of non-string keys does not break
        serialisation when exporting the contract.
        """

        return {str(key): value for key, value in self.metadata.items()}


class CustomerContract:
    """Validate and merge customer identifiers across source systems."""

    #: Fields that must be populated for the contract to be considered valid.
    REQUIRED_FIELDS = {"customer_id", "acquisition_ts", "source_system"}

    def __init__(self, enforce_visibility: bool = False) -> None:
        self.enforce_visibility = enforce_visibility

    def validate_records(
        self, records: Iterable[Mapping[str, Any]], *, source_system: str
    ) -> list[CustomerIdentifier]:
        """Validate raw records and return canonical identifiers.

        Parameters
        ----------
        records:
            Iterable of raw customer dictionaries as produced by an
            upstream system. They must at a minimum provide the required
            fields defined by :attr:`REQUIRED_FIELDS`.
        source_system:
            Name of the upstream system producing the records. This value
            is injected if individual records do not specify it.
        """

        canonical: list[CustomerIdentifier] = []
        for idx, record in enumerate(records):
            data = dict(record)  # make a mutable copy
            if "source_system" not in data or not data["source_system"]:
                data["source_system"] = source_system

            missing = [field for field in self.REQUIRED_FIELDS if not data.get(field)]
            if missing:
                raise ValueError(
                    "Record missing required contract fields",
                    {"missing_fields": missing, "record_index": idx},
                )

            acquisition_ts = data["acquisition_ts"]
            if not isinstance(acquisition_ts, datetime):
                raise TypeError(
                    "acquisition_ts must be a datetime instance",
                    {"record_index": idx, "value": acquisition_ts},
                )

            is_visible = bool(data.get("is_visible", True))
            if self.enforce_visibility and not is_visible:
                # Invisible customers are filtered unless the caller opts out.
                continue

            metadata = data.get("metadata", {})
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    "metadata must be a mapping if provided",
                    {"record_index": idx, "value": metadata},
                )

            canonical.append(
                CustomerIdentifier(
                    customer_id=str(data["customer_id"]),
                    acquisition_ts=acquisition_ts,
                    source_system=str(data["source_system"]),
                    is_visible=is_visible,
                    metadata=metadata,
                )
            )
        return canonical

    def merge(self, identifiers: Iterable[CustomerIdentifier]) -> list[CustomerIdentifier]:
        """Merge identifiers from multiple sources into canonical records.

        The merge preserves the earliest acquisition timestamp and marks
        visibility as true if the customer is visible in any upstream record.
        Metadata dictionaries are shallow-merged with later values taking
        precedence when keys overlap.
        """

        merged: dict[str, CustomerIdentifier] = {}
        for identifier in identifiers:
            existing = merged.get(identifier.customer_id)
            if existing is None:
                merged[identifier.customer_id] = identifier
                continue

            acquisition_ts = min(existing.acquisition_ts, identifier.acquisition_ts)
            is_visible = existing.is_visible or identifier.is_visible
            merged_metadata = {**existing.normalised_metadata(), **identifier.normalised_metadata()}
            merged[identifier.customer_id] = CustomerIdentifier(
                customer_id=existing.customer_id,
                acquisition_ts=acquisition_ts,
                source_system=existing.source_system,
                is_visible=is_visible,
                metadata=merged_metadata,
            )

        return list(merged.values())

    def to_serialisable(self, identifiers: Iterable[CustomerIdentifier]) -> list[dict[str, Any]]:
        """Convert identifiers into JSON-serialisable dictionaries."""

        payload: list[dict[str, Any]] = []
        for identifier in identifiers:
            payload.append(
                {
                    "customer_id": identifier.customer_id,
                    "acquisition_ts": identifier.acquisition_ts.isoformat(),
                    "source_system": identifier.source_system,
                    "is_visible": identifier.is_visible,
                    "metadata": identifier.normalised_metadata(),
                }
            )
        return payload
