# vidtight

[![CI](https://github.com/verygoodplugins/vidtight/actions/workflows/ci.yml/badge.svg)](https://github.com/verygoodplugins/vidtight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-lightgrey)

**Local screen-recording tightener.** vidtight transcribes your recording on-device with [parakeet-mlx](https://github.com/senstella/parakeet-mlx), cuts the ums, dead air, stutters and retakes, fixes mis-heard product names, and hands you an edit decision list, a retimed SRT and (optionally) a rendered MP4. No cloud upload, no timeline dragging.

## How it works

```text
recording.mov
   │
   ├─ ffmpeg ─────────────► 16 kHz mono WAV
   ├─ parakeet-mlx ───────► word-level transcript         (words.json)
   ├─ ffmpeg silencedetect ► dead-air ranges
   │
   ▼
 cuts: fillers · stutters · retakes · pauses  ──►  vocab fixes (vocab.yaml)
   │
   ▼
 EDL (edl.json) + cuts.csv  ──►  retimed captions (.srt)  ──►  --render ──► tightened .mp4
```

1. **Transcribe.** Audio is extracted and run through `parakeet-mlx` for word-level timestamps. The transcript is cached in `words.json`, so `--reuse` makes re-tuning instant.
2. **Find the cuts.**
   - **Fillers:** `um`, `uh`, `erm`, `er`, `ah`, `hmm`, `mm`, `uhm`, `umm`, plus `you know`, `kind of`, `sort of` (and `I mean` with `--cut-i-mean`). `so`, `like`, `basically`, `okay`, `ok`, `right` are cut only when they open a sentence of more than three words.
   - **Stutters:** an immediately repeated word ("the the"), except `that that` and `had had`.
   - **Retakes:** a run of at least `--retake-words` words (default 5) that is repeated within the next 12 words. The first take is dropped and the last take kept.
   - **Dead air:** ffmpeg `silencedetect` is the ground truth for pauses, because ASR word ends smear across silences. Pauses longer than `--pause` seconds (default 0.8) are cut, leaving `--pad` seconds of air (default 0.15) on each side.
3. **Fix vocabulary.** Product and brand names are corrected from [`vocab.yaml`](vocab.yaml).
4. **Build the EDL.** Cuts are inverted into keep-segments. Parakeet stamps fillers late, so filler cuts start at the audio dip before the word. Every edge snaps to the quietest frame within `--snap` seconds and is quantized to the video frame rate.
5. **Write captions.** An SRT is retimed to the tightened cut.
6. **Render (optional).** Audio is assembled sample-exactly with numpy (20 ms crossfades at every join), then one ffmpeg pass applies the optional music bed, `loudnorm` and a limiter and encodes AAC. Video is encoded in chunks with VideoToolbox hardware H.264 and muxed with the captions as a soft subtitle track.

## Requirements

- macOS on Apple Silicon (parakeet-mlx runs on MLX; rendering uses the macOS-only `h264_videotoolbox` and `aac_at` encoders)
- Python 3.11 or newer
- [ffmpeg](https://ffmpeg.org/) and `ffprobe` on your `PATH` (`brew install ffmpeg`)

Analysis (transcript, cuts, EDL, SRT) needs only the above minus the hardware encoders. The speech model (`mlx-community/parakeet-tdt-0.6b-v3`) is downloaded from Hugging Face on first use; your audio never leaves the machine.

## Installation

```bash
git clone https://github.com/verygoodplugins/vidtight
cd vidtight
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Installing in editable mode gives you the `vidtight` command and keeps the bundled `vocab.yaml` next to the script. You can also run the script directly: `python vidtight.py recording.mov`.

## Usage

```bash
# Analyze only: transcript, cuts.csv, edl.json and a retimed SRT
vidtight recording.mov

# Also render recording.vidtight/recording.mp4
vidtight recording.mov --render

# Tighter pauses, a title-card lead-in and manual edits
vidtight recording.mov --render --pause 0.6 --leadin title.mp4 --edits edits.yaml

# Re-tune cuts or vocabulary without re-running ASR
vidtight recording.mov --reuse --retake-words 4
```

Outputs land in `<source name>.vidtight/` next to the source (override with `--out`):

| File | What it is |
| --- | --- |
| `words.json` | Word-level transcript with timestamps (cache for `--reuse`) |
| `audio.wav` | 16 kHz mono audio used for ASR and silence detection |
| `edl.json` | Keep-segments `[start, end, speed]`, every cut, speed-ups and vocab fixes |
| `cuts.csv` | Every cut with start, end, type (`filler`, `stutter`, `retake`, `pause`, `manual`) and detail |
| `<name>.srt` | Captions retimed to the tightened cut |
| `<name>.mp4` | With `--render`: tightened video, AAC audio and soft subtitles (plus intermediate audio/video files) |

Review `cuts.csv` before rendering. Each row is a source timestamp range, so you can jump straight to anything suspicious.

Example: a 30-second test clip with a few ums, long pauses and one re-said sentence.

```text
$ vidtight demo.mp4 --render
cuts: {'pause': 4, 'filler': 4, 'retake': 1} speed-ups: 0 vocab fixes: 0
duration: 30.4s -> 17.3s edit (+0.0s music tail) = 17.3s (+0.0s lead-in), 7 segments
```

```csv
start,end,type,detail
00:00:02.874,00:00:05.292,pause,"2.7s"
00:00:04.910,00:00:05.920,filler,"Um,"
00:00:07.680,00:00:08.320,filler,"uh,"
00:00:14.000,00:00:17.771,retake,"the settings live in the top right corner of"
...
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `src` | required | Source video |
| `--render` | off | Render the tightened MP4 |
| `--out DIR` | `<source>.vidtight/` | Output directory |
| `--name NAME` | source file name | Output base name |
| `--pause SECONDS` | `0.8` | Cut silences longer than this |
| `--pad SECONDS` | `0.15` | Air kept around speech |
| `--mincut SECONDS` | `0.25` | Skip pause cuts shorter than this after padding |
| `--noise DB` | `-35` | Silence threshold |
| `--retake-words N` | `5` | Minimum repeated words to call a retake |
| `--cut-i-mean` | off | Also cut "I mean" as a filler |
| `--snap SECONDS` | `0.08` | Edge snap window to the quietest frame (`0` = off) |
| `--vocab FILE` | `vocab.yaml` beside the script | Vocabulary corrections |
| `--edits FILE` | none | Manual edits (see below) |
| `--model ID` | `mlx-community/parakeet-tdt-0.6b-v3` | Parakeet model |
| `--reuse` | off | Reuse the cached `words.json` |
| `--bitrate RATE` | `16M` | Video bitrate for the render |
| `--leadin FILE` | none | Pre-encoded title-card clip (same size, fps and codec) to prepend |
| `--fadein SECONDS` | `0` | Fade the first footage in from black |

## Custom vocabulary

`vocab.yaml` maps each canonical term to the ways the ASR tends to mishear it:

```yaml
# canonical: [misheard variants]
WP Fusion: [WP fusion, W P Fusion, WP Fushion]
Claude Code: [Cloud Code, Clod Code]
```

Matching ignores case and punctuation and preserves trailing punctuation and possessives. A fuzzy pass also catches near misses of the canonical terms themselves (same word count, at least six characters, similarity of 0.85 or more). Keep a private list in `vocab.personal.yaml` (git-ignored) and pass it with `--vocab vocab.personal.yaml`.

## Manual edits

`--edits edits.yaml` layers human judgment on top of the automatic cuts. All times are source-video seconds.

```yaml
keep_retakes: [14.0]            # keep an auto-detected retake that starts within 2 s of this time
speed:                          # play the pause containing this time at Nx (muted) instead of cutting it
  - [22.5, 4]
cut:                            # extra cuts: [start, end, reason]
  - [30.0, 34.5, "tangent"]
replace:                        # text replacements applied to caption lines
  "teh": "the"
music:                          # optional music bed
  file: bed.mp3
  start_src: 0.0                # source time where the music enters
  slam_src: 120.0               # source time where it rises from the ducked level to full
  duck_db: -19
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest          # unit tests for the pipeline logic (no media or Apple Silicon needed)
ruff check .    # lint
```

`vidtight.py` is a single module written in a deliberately compact style, so `ruff format` is not enforced. See [AGENTS.md](AGENTS.md) for contributor and agent conventions and [CHANGELOG.md](CHANGELOG.md) for release notes.

## Support

For issues, questions, or suggestions:

- [Open an issue on GitHub](https://github.com/verygoodplugins/vidtight/issues)
- [Contact Very Good Plugins](https://verygoodplugins.com/contact/?utm_source=github)

---

Built with 🧡 by [Very Good Plugins](https://verygoodplugins.com/?utm_source=github)
