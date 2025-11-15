## Contract v1 Overview

All first-class content models inherit from `Artifact`:

- `kind` — machine label like `explanation.v1`, `term.v1`, …
- `version` — semantic schema version (default `1.0.0` for v1)
- `confidence` — calibrated score in `[0, 1]`
- `provenance[]` — traceability entries (`source`, `locator`, `model`, `created_at`, `note`)

### Models

- `ExplanationCard` — `{ claim, rationale, evidence[], summary? }`
- `TermCard` — `{ term, definition, aliases[], examples[], sources[] }`
- `RelevanceNote` — `{ target, rationale, score }`
- `TimelineEvent` — `{ when, title, description?, tags[], sources[] }`
- `PublicBrief` — `{ title, summary, sections[] }` with `BriefSection { heading, body, bullets[] }`
- `ReviewCriteria` — `{ accuracy, clarity, completeness, safety }` (+ artifact fields)
- `ReviewReport` — `{ overall, criteria, comments[], actions[] }`

### Versioning Rules

- Backward-compatible additive fields: bump **minor** (e.g., `1.1.0`).
- Breaking changes (rename/semantics): bump **major** (e.g., `2.0.0`).
- Bugfixes/description-only tweaks: bump **patch** (e.g., `1.0.1`).

### JSON Schemas

JSON Schema files live under `schemas/`:

- `explanation.v1.json`
- `term.v1.json`
- `relevance.v1.json`
- `timeline_event.v1.json`
- `public_brief.v1.json`
- `review_criteria.v1.json`
- `review_report.v1.json`

These are validated in `tests/test_contracts_schema.py` by comparing generated schema
titles, required sets, and representative properties.
