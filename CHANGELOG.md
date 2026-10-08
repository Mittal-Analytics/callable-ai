# Changelog

## 0.2.0 - 2026-10-08

### Breaking

- `AIModel.input_tokens_cache_write_cost_usd` is now required. Anthropic and OpenAI GPT-5.6+ bill cache writes at 1.25x input; use the input price for providers without a write premium.

### Changed

- Enabled OpenRouter's automatic prompt caching on all requests. Anthropic models only cache marked prompts, so tool-call turns now reread history at the cached input price.
- Costing bills OpenRouter's `cache_write_tokens` at the cache-write price, on both APIs.

## 0.1.10 - 2026-09-17

### Changed

- Required OpenAI SDK 3.14.1 or newer (below 4), using HTTPX2 and the operating-system certificate store.
- Replaced private parsing helpers with the official `responses.parse`, which skips explicitly marked commentary.
- Structured-output validation errors now occur before the failed response is appended to history.

## 0.1.9 - 2026-09-17

### Fixed

- Deferred structured-response parsing until tool calls finish, allowing plain-text commentary on tool turns.
- Retained model output in history when structured-output validation fails, allowing corrective follow-ups.

## 0.1.8 - 2026-09-04

### Fixed

- Emitted only the final cumulative usage reported by streaming providers.

## 0.1.7 - 2026-09-04

### Fixed

- Preserved OpenRouter Gemini signature-only reasoning details without emitting malformed text events.
- Removed unsupported request fields from Google OpenAI-compatible streaming calls.

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
