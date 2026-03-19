# Contributing

## Workflow

1. Create a feature branch.
2. Make focused changes.
3. Verify notebooks run top-to-bottom.
4. Open a pull request with a short summary of results.

## Notebook Conventions

- Keep phase order: extraction -> preprocessing -> model.
- Add markdown cells for any major logic changes.
- Avoid committing large generated outputs unless needed for documentation.

## Code Style

- Prefer clear, readable Python.
- Keep physics-loss logic in `physics_model.py`.
- Keep notebook cells modular and rerunnable.
