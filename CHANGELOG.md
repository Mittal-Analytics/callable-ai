# Changelog

## 0.1.6 - 2026-09-01

### Changed

- Replaced the streaming `user` argument with a required `prompt_cache_key` across all response entry points.
- Reused `prompt_cache_key` as the OpenRouter session ID for sticky provider routing across tool-call follow-ups.
- Removed the OpenRouter `middle-out` message transform.

## 0.1.5 - 2026-09-01

### Changed

- Renamed the distribution from `mittal-ai` to `callable-ai`.
- Renamed the Python package from `mittal_ai` to `callable_ai`.

## 0.1.4 - 2026-09-01

### Added

- Added typed callable tools that may return directly, await a result, or stream progress events.
- Added normalized tool-call events for streaming responses.

### Changed

- Tool calls now run concurrently while forwarding progress events.
- Interrupted responses now cancel running tools and repair their message history.
- Removed the Django and `dj-evals` runtime dependencies.

## 0.1.3 - 2026-08-21

### Fixed

- Preserved nested argument details when parsing multiline tool docstrings.

## 0.1.2 - 2026-08-21

### Changed

- Lowered the minimum supported Django version to 5.1 and removed the upper bound.
- Updated the pre-commit hooks.

## 0.1.1 - 2026-08-21

### Added

- Added trusted publishing through GitHub Actions.
- Documented the PyPI release process.
