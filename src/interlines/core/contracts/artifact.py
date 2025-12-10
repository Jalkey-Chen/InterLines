"""Base artifact contracts shared by higher-level content models.

This module defines two Pydantic v2 models:

- `Provenance`: where a content fragment came from (source, locator, model, time).
- `Artifact`  : a typed envelope embedded by all first-class content models,
  carrying `kind`, semantic `version`, and a calibrated `confidence` score.

Versioning
----------
We use a semantic *schema* version string in `version` (e.g., "1.0.0").
Model-specific files will start at "1.0.0" and increment when we add, deprecate,
or change field meanings. Backward-compatible additive changes bump the *minor*
version; breaking changes bump the *major* version.

Confidence
----------
`confidence` is a calibrated score in [0.0, 1.0]. Producers should document how
it is computed (e.g., rubric mapping, agreement rate, evaluator model score).

Notes
-----
- Keep models conservative and explicit; downstream renderers rely on these shapes.
- Pydantic v2 is used (see pyproject). We validate ranges and simple patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

# ---- Shared small types ------------------------------------------------------

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Semver = Annotated[
    str,
    Field(
        pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z\.-]+)?$",
        description="Semantic version (MAJOR.MINOR.PATCH), optional pre-release/build.",
    ),
]


class Provenance(BaseModel):
    """Traceability metadata indicating where an artifact element came from."""

    source: Literal["user", "web", "pdf", "dataset", "model", "other"] = Field(
        description="High-level origin category."
    )
    locator: str | HttpUrl | None = Field(default=None, description="URL or opaque locator")
    model: str | None = Field(default=None, description="Producer model/agent id, if any")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this fragment was created/observed.",
    )
    note: str | None = Field(default=None, description="Freeform provenance note")


class Artifact(BaseModel):
    """Base envelope embedded by all content contracts."""

    # REQUIRED (no defaults) to match schemas/*.json
    kind: str = Field(description="Short machine label, e.g. 'explanation.v1'")
    version: Semver = Field(description="Schema version (semver)")
    confidence: Confidence = Field(description="Calibrated score in [0,1]")

    provenance: list[Provenance] = Field(default_factory=list, description="Traceability entries")

    @field_validator("kind")
    @classmethod
    def _must_have_dot(cls, v: str) -> str:
        """Encourage a `<name>.v<major>` style for portability."""
        if "." not in v:
            raise ValueError("kind should include a dotted suffix, e.g., 'explanation.v1'")
        return v

    @model_validator(mode="after")
    def _normalize(self) -> Artifact:
        """Optional post-init normalization hook (reserved for future use)."""
        return self


__all__ = ["Artifact", "Provenance", "Confidence", "Semver"]
