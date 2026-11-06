# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
