# Contributing

## Development setup

Create a Python 3.10 or newer environment and install the project with its test dependencies:

```bash
python -m pip install -e ".[test]"
```

## Validation

Run the canonical test suite before opening a pull request:

```bash
./scripts/run_tests.sh
python -m compileall -q .
python -m build
```

The test runner executes every `test_*.py` script followed by `validate_api.py`. Keep tests deterministic and avoid requiring external credentials or hosted infrastructure.

## Pull requests

Keep changes focused, explain runtime or persistence implications, and include tests for new endpoints, state transitions, and failure paths. Changes to neural dynamics should document any new configuration fields, loss terms, checkpoint fields, or public projection changes.

Do not commit credentials, generated model checkpoints, runtime state, corpus caches, or deployment-specific configuration. Use environment variables and update `.env.example` when adding a new setting.
