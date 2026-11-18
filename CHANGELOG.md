# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `--select moments.yaml`: moment-selection mode. Keeps a window around each
  listed moment (`{t, label, pre?, post?, caption?}`) and puts every edge in a
  speech gap, then renders through the existing path. An edge that lands in
  speech moves outward to take the line whole (up to `--max-clip`) or inward to
  the nearest gap. A moment that fits neither way is dropped and reported.
- Edge rules for `--select`: never inside a word, gaps of 0.4 s or more
  preferred (`--gap`), 0.15 to 0.4 s gaps only at sentence ends (`--min-gap`),
  pads of 30-200 ms that snapping and frame rounding cannot eat.
- `--speech FILE`: exact `[start, end, label]` speech intervals from the app.
  They replace the ASR words they overlap.
- `--words FILE`: use an existing transcript instead of running ASR.
- `--chunk SECONDS`: ASR chunk length. Defaults to 30 with `--select`, because
  120 s chunks dropped 12 whole lines (62 of 169 words) on a recording with
  music under the narration. The tighten mode keeps 120.
- Speech map that survives smeared ASR timings: word ends are cut back to the
  first detected silence, else capped at 0.3 s + 0.1 s per letter
  (`--max-word`). The report says `LOW CONFIDENCE` when word ends are estimates
  and warns when a word is stamped 3 s or longer (lines may be missing).
- `select.csv` (one row per moment: status, in, out, whether each edge is
  speech-safe and why) and extra `select` / `speech` keys in `edl.json`.
- Self-check: the EDL is read back from disk and every edge is checked against
  the speech map. `--strict` exits 2 if any edge is unsafe.
- `speech_blocks`, `speech_gaps`, `select_moments` and `check_edl` are plain
  functions, usable as a library step. `tests/test_select.py` covers them with
  synthetic word lists (no ASR), including a 150-transcript random sweep.

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

- Variable-frame-rate captures that report a tick rate as `r_frame_rate`
  (Quest: `100000/1`) were put on a 100000 fps grid. The average frame rate is
  used for those.
- The final mux used `-shortest`, so the render stopped at the last caption and
  lost any footage after the last spoken word (a 47.5 s cut came out at 28.2 s).
  The mux now ends with the video.
- `--reuse` into a fresh `--out` directory crashed on the missing `audio.wav`.
  It is extracted again when absent.
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
