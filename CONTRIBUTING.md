# Contributing

Issues and pull requests are welcome. This is a small benchmark, so the bar
for changes is about comparability more than anything else.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install
make check
```

`make check` is exactly what CI runs: `ruff`, `mypy --strict`, and `pytest`
with a 90% coverage floor. No API key is needed; the end-to-end tests use the
scripted reference policies.

## Rules that keep results comparable

1. **Anything that changes a score needs a version bump.** If a change alters
   generated scenarios, a prompt, a tool, or the scorer, bump
   `SCHEMA_VERSION` in `scenario.py` (which feeds the task version) and say
   so in the pull request. Results from different versions must not be mixed
   in one table.
2. **The frozen dataset is the source of truth.** Do not regenerate
   `src/memorydrift/data/scenarios_v1.jsonl` in place. Add a new file.
3. **Reference scores are derived, then asserted.** If you change the
   blueprint, re-derive the expected scores in `tests/test_end_to_end.py` by
   hand and explain the derivation in `docs/design.md`. Do not paste in
   whatever the run printed.
4. **Prompts live in `prompts.py`.** They are part of the method, so they are
   kept in one place and are not tuned per model.
5. **A sample error is never a failure.** New analysis code must keep
   excluded samples out of every rate and report their count.

## Style

Google Python style: docstrings with `Args`/`Returns`/`Raises`, type
annotations on everything, 80 columns. `make format` applies it.

## Reporting a result

Include the model and its version, the date, the number of scenarios and
epochs, the `inspect-ai` version, and the excluded-sample count. The
`results.json` written by `memorydrift report` carries all of these except
the model version, which the provider decides.
