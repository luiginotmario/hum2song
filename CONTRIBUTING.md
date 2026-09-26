# Contributing

Phase 1 code lives under `training/`. These rules apply to every Python file in the repo.

- Imports first, then module constants, then every function and class. The only executable entry is `if __name__ == "__main__": main()` at the bottom of a script. Do not define a function below code that runs.
- Keep functions small. Shared behavior (audio loading, manifest IO, config, seeding, logging) belongs in one module. Do not copy it into a second script.
- Use descriptive names, type hints, and a short docstring on each public function and class.
- Prefer early returns, lookup tables, and short conditional expressions. Do not write nested `if`/`else` chains.
- Format and lint with [ruff](ruff.toml) before pushing: `ruff check .` and `ruff format .`.
- Do not add license or copyright headers to source files.
- Do not commit audio, checkpoints, `.env`, or dataset archives. `data/`, `runs/`, and `ckpt/` are gitignored.

CPU check, no GPU required:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]" numpy PyYAML soundfile
ruff check .
ruff format --check .
pytest
```
