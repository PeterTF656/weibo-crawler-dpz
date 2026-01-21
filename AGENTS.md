# Repository Guidelines

## Project Structure & Module Organization

- `weibo.py`: main crawler (the `Weibo` class) and one-off run entry.
- `__main__.py`: scheduled runner that calls `weibo.main()` every *N* minutes.
- `service.py`: Flask API wrapper around the crawler + SQLite read endpoints (see `API.md`).
- `util/`: helper modules (`csvutil.py`, `dateutil.py`, `notify.py`, `llm_analyzer.py`).
- Config/logging: `config.json` (copy from `config.json.example`) and `logging.conf`.
- Generated output: `weibo/` (data/media) and `log/` are created at runtime and are gitignored.

## Build, Test, and Development Commands

- Install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Create local config: `cp config.json.example config.json`
- Run crawler once: `python weibo.py`
- Run on a schedule (minutes): `python __main__.py 1`
- Run API server: `python service.py`
- Docker (mounts `./config.json`, optional `./user_id_list.txt`, and `./weibo/`): `docker compose up --build -d`

## Coding Style & Naming Conventions

- Python only; use 4-space indentation and keep formatting consistent with nearby code (avoid sweeping reformatting of `weibo.py`).
- Prefer `snake_case` for functions/variables and `UPPER_SNAKE_CASE` for constants.
- Use the configured `logger` (via `logging.conf`) instead of ad-hoc `print()` for runtime output.

## Testing Guidelines

- No dedicated automated test suite. For quick checks: `python -m compileall .`.
- `test_llm.py` is a manual smoke script for `util/llm_analyzer.py` and requires `llm_config` in `config.json`.

## Commit & Pull Request Guidelines

- Match the repo’s common convention: short Conventional Commits-style subjects like `feat: ...`, `fix: ...`, `docs: ...`, optionally with `Issue #NNN`.
- PRs should include: rationale, how you tested (commands), any new `config.json` keys (also update `config.json.example`), and doc updates (`README.md`/`API.md`) when behavior changes.

## Security & Configuration Tips

- Never commit real cookies, tokens, or LLM API keys. Prefer the `WEIBO_COOKIE` env var over hardcoding cookies in `config.json`.
- Keep user-specific files (e.g. `user_id_list.txt` and output under `weibo/`) out of commits.
