# vidtight

[![CI](https://github.com/verygoodplugins/vidtight/actions/workflows/ci.yml/badge.svg)](https://github.com/verygoodplugins/vidtight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-lightgrey)

**Local screen-recording tightener.** vidtight transcribes your recording on-device with [parakeet-mlx](https://github.com/senstella/parakeet-mlx), cuts the ums, dead air, stutters and retakes, fixes mis-heard product names, and hands you an edit decision list, a retimed SRT and (optionally) a rendered MP4. No cloud upload, no timeline dragging.

It also runs the other way round. [`--select`](#selecting-moments---select) keeps only the moments you list and never cuts through a spoken line.

<video src="https://github.com/user-attachments/assets/60ecdbd2-0931-40e6-919a-03a2b780f4f3"></video>

<p align="center">
  <sub>Overview of vidtight, rendered with Remotion using simulated sample data and generated background music.</sub>
</p>

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

Installing in editable mode gives you the `vidtight` command and keeps the bundled `vocab.yaml` next to the script. A regular (non-editable) install ships only `vidtight.py`, so it runs without the default vocabulary; pass your own with `--vocab`. You can also run the script directly: `python vidtight.py recording.mov`.

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

# Keep only the listed moments, every edge in a speech gap (see "Selecting moments")
vidtight capture.mp4 --select moments.yaml --render
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
| `--vocab FILE` | `vocab.yaml` beside the script, if present | Vocabulary corrections |
| `--edits FILE` | none | Manual edits (see below) |
| `--model ID` | `mlx-community/parakeet-tdt-0.6b-v3` | Parakeet model |
| `--reuse` | off | Reuse the cached `words.json` |
| `--words FILE` | none | Use this `words.json` instead of running ASR |
| `--chunk SECONDS` | `120`, or `30` with `--select` | ASR chunk length. Long chunks drop whole lines when music runs under the voice |
| `--select FILE` | none | Moments to keep (see [Selecting moments](#selecting-moments---select)). The options below apply to this mode |
| `--speech FILE` | none | Exact `[start, end, label]` speech intervals from the app. They replace overlapping ASR words |
| `--pre SECONDS` / `--post SECONDS` | `1` / `4` | Window kept before and after each moment |
| `--max-clip SECONDS` | `12` | Longest a clip may grow to take a spoken line whole |
| `--gap SECONDS` | `0.4` | Smallest speech gap an edge may use |
| `--min-gap SECONDS` | `0.15` | Smallest gap usable between two exact `--speech` intervals |
| `--max-word SECONDS` | `1.2` | Longest a single word can last. Longer ASR stamps are cut back |
| `--strict` | off | Exit 2 if any edge of the final EDL is not speech-safe |
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

## Selecting moments (`--select`)

The tighten mode removes what you don't want. `--select` works the other way round: hand it a list of moments and it keeps a window around each one, puts every edge in a gap between spoken lines, and renders through the same path (edge snap, frame grid, sample-exact audio joins).

It exists because of a trailer that was cut on gameplay timestamps alone. Half the window edges landed inside narration.

```bash
vidtight capture.mp4 --select moments.yaml --out /tmp/capture.select --render
```

```yaml
# moments.yaml: times are source seconds
- {t: 14.5, label: "room asleep, title"}
- {t: 24.4, label: "takes an orb", pre: 0.5, post: 3}    # per-moment window
- {t: 109.8, label: "supernova", caption: "Supernova"}    # caption lands in the SRT over that clip
```

Each moment asks for `t - pre` to `t + post` (`--pre 1`, `--post 4` unless the moment says otherwise).

### Where the edges go

1. **Speech map.** Built from the word-level transcript. With `--select` the ASR runs in 30 s chunks (`--chunk`) instead of 120 s. On a 160 s Quest capture with music under the narration, 120 s chunks heard 107 words and dropped 12 whole lines. 30 s chunks heard 169.
2. **Word ends.** Parakeet stamps a word until the next one starts, so the last word of a line swallows the pause after it. Each word end is cut back to the first `silencedetect` pause inside it. With music under the voice there is no such pause, and the word is capped at 0.3 s plus 0.1 s per letter (never more than `--max-word`). A capped end is an estimate, and the report counts them.
3. **App-logged speech.** `--speech FILE` takes exact `[start, end, label]` intervals (YAML or JSON) from an app that logs its own voice lines. They replace the ASR words they overlap and are never cut.
4. **Gaps.** 0.4 s or more (`--gap`) is a clean gap. A tight gap of 0.15 to 0.4 s (`--min-gap`) only counts between two `--speech` intervals, because ASR timings can't place one. On the test capture, parakeet's word starts ran from 0.39 s early to 0.17 s late against the app's own voice-line times, and a 0.26 s gap placed from them put an edge 16 ms into the next line. Between exact intervals a tight gap is still taken only when it saves more than a second of edge movement over a clean one.
5. **Moving an edge.** An edge that lands in speech moves to a gap. Outward takes the line whole, and the clip may grow to `--max-clip`. Inward trims the window, never below half of what was asked for. The cheapest pair of moves wins. When nothing fits, the moment is dropped and the report says why.
6. **Pads.** An edge sits `--pad` after a word (0.15 s, clamped to 30-200 ms) and 200 ms before one, since parakeet stamps word starts late. A gap too narrow for both pads splits the room pro rata, never under 30 ms a side. Snapping still looks for the quietest frame, but it can't leave the gap or eat a pad.

Clips closer than 0.5 s are merged, and clips always come out in source order.

### What you get

| File | What it is |
| --- | --- |
| `edl.json` | Same schema as the tighten mode. `keep` holds the clips and `cuts` the unselected ranges, plus `select` (one entry per moment) and `speech` (the utterances) |
| `select.csv` | One row per moment: status (`kept`, `widened`, `trimmed`, `dropped`, `+merged`), clip number, in, out, whether each edge is speech-safe and why, what moved, and the speech the clip contains |
| `<name>.srt` | The spoken words inside the clips plus any moment captions, retimed |

After writing `edl.json`, vidtight reads it back and checks every edge against the speech map:

```text
select: 9 moments -> 8 clips (3 as asked, 6 widened, 0 trimmed, 2 merged, 0 dropped)
as asked, 9 of 18 window edges were inside speech
self-check: 16 edges re-read from edl.json: 0 inside speech, 0 under 30 ms from a word, 0 in tight gaps (under 0.4s), 8 rest on an estimated word end
```

The first two numbers should be zero. If they aren't, a warning goes to stderr, and with `--strict` vidtight exits 2 before rendering.

### Limits

- **The check is only as good as the transcript.** An edge inside speech the ASR never heard passes it. When the report says `LOW CONFIDENCE`, listen to the cut, or log the lines in the app and pass `--speech`.
- Music or room tone under the voice hides the pauses from `silencedetect`, so word ends there are estimates from the length cap.
- ASR-only cuts run long, because only gaps of 0.4 s or more count. The same nine moments gave 58.5 s from ASR alone and 44.2 s with the app's exact `--speech` intervals.
- Clips play in source order. Reordering for story is the caller's job.
- The recording's own music is cut with the picture, so a bed jumps at every join.
- No filler or pause cuts run inside the clips, and `--edits` only applies `replace` and `music`.
- A moment that sits mid-line gets the whole line or is dropped. Raise `--max-clip` to keep long lines.

### As a library step

The pieces are plain functions, so a pipeline can call them without the CLI:

```python
import vidtight as v

blocks = v.speech_blocks(words, speech=app_lines)            # words.json list, [[start, end, label], ...]
gaps = v.speech_gaps(blocks, dur)
clips, rows = v.select_moments(moments, gaps, blocks, dur)   # [[in, out], ...] and one dict per moment
unsafe = [e for e in v.check_edl([[i, o, 1] for i, o in clips], blocks, dur) if not e["safe"]]
```

`check_edl` takes any keep list, so it can also vet an EDL another tool made.

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
