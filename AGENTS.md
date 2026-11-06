# AGENTS.md

## Repository purpose

`vidtight` is a local, single-module CLI (`vidtight.py`) that tightens screen
recordings: parakeet-mlx ASR → filler/stutter/retake/pause cuts → vocabulary
fix → EDL + SRT + ffmpeg render. It runs on macOS / Apple Silicon.

## Common commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python vidtight.py --help
pytest
ruff check .
```

## Project conventions

- Target Python 3.11 or newer.
- Keep the project a single module. The compact one-step-per-line style in
  `vidtight.py` is deliberate: `ruff check` is configured for it and
  `ruff format` is not enforced. Do not mass-reformat.
- `parakeet_mlx` is imported lazily inside `transcribe()` so the module and the
  tests import on machines without Apple Silicon. Keep it lazy.
- Rendering uses macOS-only encoders (`h264_videotoolbox`, `aac_at`). Analysis
  (transcript, cuts, EDL, SRT) needs only ffmpeg/ffprobe plus parakeet-mlx.
- Never parse tool output with `eval`; use `fractions.Fraction` or `json`.
- `vocab.yaml` is a public default holding product/brand terms only. Personal
  names and private project names go in the git-ignored `vocab.personal.yaml`
  (pass it with `--vocab`).
- Never commit media, run directories, virtualenvs, transcripts or generated
  outputs. They routinely contain absolute local paths and private speech.
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:` ...). PR titles
  are linted.
- Before opening a change, run `pytest` and `ruff check .`.

## Layout

```text
vidtight.py     CLI + pipeline (ASR, cuts, EDL, SRT, render)
vocab.yaml      Default vocabulary corrections (brand/product terms)
tests/          Unit tests for the pure pipeline logic
.github/        CI, security, PR-title and Dependabot automation
```

## Releases

Releases are tag-driven, not managed by Release Please. Update `CHANGELOG.md`,
then create and push a version tag such as `v0.1.0`.
