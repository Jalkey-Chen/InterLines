# Prompt Suite for InterLines

This document describes how we structure, name, and version prompts in the
InterLines / PKI codebase. The goal is to make prompts:

- **Discoverable** — easy to find by agent / task.
- **Composable** — shared building blocks across agents (e.g. tone rules).
- **Testable** — stable enough that we can write unit tests around them.
- **Provider-agnostic** — prompts describe *intent*; the `LLMClient` and
  `ModelConfig` decide which concrete model to use.

---

## 1. Filesystem layout

Prompts live close to the LLM client:

- Code: `src/interlines/llm/prompts.py` (and possibly submodules later)
- Docs: `docs/PROMPTS.md` (this file)

Later, if the prompt suite grows large, we may split by domain:

- `src/interlines/llm/prompts/core.py` — shared helpers and base templates
- `src/interlines/llm/prompts/agents.py` — agent-specific templates
- `src/interlines/llm/prompts/debug.py` — debugging / inspection prompts

For now, a single `prompts.py` module with a small number of functions is
sufficient.

---

## 2. Design goals

Each prompt template should:

1. **Be explicit about its role.**

   We distinguish at least three roles:

   - `system` — global constraints, persona, safety, output format rules.
   - `user` — the main task description or question.
   - `assistant` — examples of good behaviour (few-shot demonstrations).

2. **Be minimal and layered.**

   - A core system prompt defines *general* behaviour (e.g. “you are a
     careful explainer for public readers”).
   - Narrower functions add task-specific instructions (e.g. "produce a
     one-sentence gist", "write in Markdown with H2 headings").

3. **Expose a typed API.**

   Prompt builders should return `list[dict[str, str]]` in the same shape
   expected by `LLMClient.generate()`:

   ```python
   messages = [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "..."},
   ]
    ```

4. **Keep placeholders explicit.**

   Dynamic content should use `{{snake_case_placeholders}}` rather than
   string concatenation. The outer code is responsible for filling the
   placeholders before passing the messages to the client.

---

## 3. Naming conventions

We use **agent-scoped, task-oriented** names for prompt builders.

Examples (not yet implemented in code, but reserved by convention):

* `build_parser_prompt(...)`
* `build_explainer_one_sentence_prompt(...)`
* `build_explainer_three_paragraph_prompt(...)`
* `build_explainer_deep_dive_prompt(...)`
* `build_citizen_relevance_prompt(...)`
* `build_jargon_term_card_prompt(...)`
* `build_history_timeline_prompt(...)`
* `build_editor_review_prompt(...)`
* `build_brief_markdown_prompt(...)`

The naming rule is:

```text
build_<agent>_<task>_prompt
```

Where:

* `<agent>` is one of: `parser`, `explainer`, `citizen`, `jargon`,
  `history`, `editor`, `brief`, etc.
* `<task>` describes the granularity: `one_sentence`, `deep_dive`,
  `timeline`, `review`, `markdown`, …

This makes it easy to grep for the exact prompt used by a given agent.

---

## 4. Versioning and stability

Prompts are versioned *informally* via comments and commit messages, and
*formally* via the contracts when necessary:

* If a prompt change also changes the expected output schema, the contract
  kind/version (e.g. `explanation.v1` → `explanation.v2`) should be updated
  and reflected in the Pydantic models and JSON schemas.
* If a prompt change only affects *style* (tone, examples) but not the
  schema, we keep the same contract version and note the change in the PR
  description.

We avoid embedding version numbers into the prompt text itself; instead we
rely on:

* Git history,
* contract versions (`kind`, `version` fields),
* and trace snapshots (`artifacts/trace/`).

---

## 5. Example prompt structure (explainer)

A typical explainer prompt for the `explainer` agent might look like:

```python
def build_explainer_three_paragraph_prompt(seed_text: str) -> list[dict[str, str]]:
    system = (
        "You are an expert explainer. Your job is to rewrite complex, "
        "expert-facing text so that an informed but non-specialist reader can "
        "understand the main ideas without losing important nuance. "
        "Write in clear, structured paragraphs."
    )

    user = (
        "Read the following text and write a three-paragraph explanation. "
        "Do not invent new facts; if something is unclear, say that it is "
        "unclear.\n\n"
        f"TEXT START:\n{seed_text}\nTEXT END."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
```

Later, the explainer agent would do something like:

```python
from interlines.llm.client import LLMClient
from interlines.llm.prompts import build_explainer_three_paragraph_prompt

client = LLMClient.from_env(default_model_alias="explainer")
messages = build_explainer_three_paragraph_prompt(seed_text)
text = client.generate(messages)
```

---

## 6. Safety and tone

All prompts should:

* Avoid asking the model to *invent* facts; instead, encourage explicit
  statements like “the source text does not say …”.
* Be transparent about uncertainty and partial evidence.
* Use respectful, non-alarmist language, especially in public-facing
  prompts (`citizen` / `brief` agents).

When in doubt, default to a more conservative prompt and add clarifying
instructions rather than making the model “more creative”.

---

## 7. Testing prompts

Even though prompts are text, we should still test their **structure**:

* Unit tests can assert that prompt builders:

  * return a non-empty list of messages,
  * contain at least one `system` and one `user` message,
  * interpolate required placeholders without leaving `{{…}}` dangling
    (unless intentionally left for a higher-level template).

Later, when we introduce golden tests for specific outputs, they should be
guarded carefully (with generous tolerances) to avoid flakiness when model
behaviour changes slightly.