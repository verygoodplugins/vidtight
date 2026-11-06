# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `release.yml`: tagging `vX.Y.Z` verifies the version, builds the sdist and
  wheel and creates a GitHub Release (no PyPI publish).
- Tests for the I/O edges (ffprobe/silencedetect parsing, ASR token merging, RMS
  loading, edge snapping) and for the analysis-only CLI run, taking line coverage
  above the 50% ecosystem minimum.
- `.gitignore` entries for `.env*`, `*.v1.*` and generated transcripts, edit
  lists and captions (`words.json`, `edl.json`, `cuts.csv`, `*.srt`).

### Changed

- `pyproject.toml` keywords widened for discoverability.

### Fixed

- A non-editable install (for example `pip install git+https://...`) crashed
  because the default `vocab.yaml` is not packaged. The default is now optional;
  an explicit `--vocab` that does not exist is still an error.
- An empty vocabulary (for example an empty `--vocab` file) crashed the fuzzy
  pass in `apply_vocab` with `max() iterable argument is empty`.

## [0.1.0] - 2026-11-06

### Added

- Initial public release of `vidtight`.
- Local ASR with `parakeet-mlx` (word-level timestamps), cached as `words.json`.
- Cuts for filler words, stutters, retakes and dead air, with edge snapping to
  the quietest nearby audio frame.
- Vocabulary correction from `vocab.yaml` (explicit variants plus fuzzy match).
- Outputs: `edl.json`, `cuts.csv`, retimed `<name>.srt`.
- `--render` to produce a tightened MP4 (numpy-assembled audio with crossfades,
  loudness normalisation, chunked VideoToolbox video encode, soft subtitles).
- `--edits` file for manual cuts, kept retakes, speed-ups, caption replacements
  and a music bed.

### Security

- Parse the ffprobe frame rate with `fractions.Fraction` instead of `eval`.
