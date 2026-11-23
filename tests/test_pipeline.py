"""Tests for the I/O edges of vidtight.py: ffmpeg wrappers, ASR token merging, RMS loading and the analysis-only CLI run.

ffmpeg, ffprobe and parakeet-mlx are all stubbed, so these run on any machine without media or Apple Silicon.
"""
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vidtight as v


def spoken(*items):
    """Word dicts spaced 0.5s apart; a trailing '.' marks end of sentence."""
    out = []
    for i, w in enumerate(items):
        d = {"w": w, "s": i * 0.5, "e": i * 0.5 + 0.4}
        if w.endswith("."):
            d["eos"] = True
        out.append(d)
    return out


def write_wav(path, samples):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


# --- ffmpeg / ffprobe wrappers ---------------------------------------------------------------


def test_duration_parses_ffprobe_output_and_raises_on_failure(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, kw=kw)
        return SimpleNamespace(stdout="12.5\n", stderr="")

    monkeypatch.setattr(v.subprocess, "run", fake_run)
    assert v.duration("clip.mov") == pytest.approx(12.5)
    assert seen["cmd"][0] == "ffprobe" and seen["cmd"][-1] == "clip.mov"
    assert seen["kw"]["check"] is True  # a failing tool must raise, not be swallowed


def test_silences_parses_silencedetect_output(monkeypatch):
    err = (
        "[silencedetect @ 0x1] silence_start: 1.5\n"
        "[silencedetect @ 0x1] silence_end: 3.25 | silence_duration: 1.75\n"
        "[silencedetect @ 0x1] silence_start: 8.0\n"  # recording ends in silence: no silence_end
    )
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return SimpleNamespace(stdout="", stderr=err)

    monkeypatch.setattr(v.subprocess, "run", fake_run)
    assert v.silences("clip.mov", -35, 0.8) == [(1.5, 3.25), (8.0, None)]
    assert "silencedetect=n=-35dB:d=0.8" in seen["cmd"]


# --- ASR ---------------------------------------------------------------------------------------


def test_transcribe_merges_subword_tokens_into_words(monkeypatch, tmp_path):
    tok = lambda text, s, e: SimpleNamespace(text=text, start=s, end=e)  # noqa: E731
    sentences = [
        SimpleNamespace(tokens=[tok(" Hel", 0.0, 0.2), tok("lo", 0.2, 0.4), tok(" there.", 0.5, 0.9)]),
        SimpleNamespace(tokens=[tok("Okay", 1.0, 1.3), tok(" go", 1.4, 1.6)]),  # new sentence starts a new word without a leading space
    ]
    seen = {}

    class Model:
        def transcribe(self, wav, **kw):
            seen.update(wav=wav, kw=kw)
            return SimpleNamespace(sentences=sentences)

    def from_pretrained(model):
        seen["model"] = model
        return Model()

    monkeypatch.setitem(sys.modules, "parakeet_mlx", SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setattr(v, "sh", lambda cmd: "")  # skip the ffmpeg audio extraction
    words, took = v.transcribe("clip.mov", tmp_path, "some/model")
    assert words == [
        {"w": "Hello", "s": 0.0, "e": 0.4},
        {"w": "there.", "s": 0.5, "e": 0.9, "eos": True},
        {"w": "Okay", "s": 1.0, "e": 1.3},
        {"w": "go", "s": 1.4, "e": 1.6, "eos": True},
    ]
    assert took >= 0
    assert seen["model"] == "some/model"
    assert seen["wav"] == str(tmp_path / "audio.wav")


# --- audio analysis ----------------------------------------------------------------------------


def test_load_rms_returns_one_frame_per_10ms(tmp_path):
    write_wav(tmp_path / "a.wav", np.concatenate([np.zeros(8000), np.full(8000, 10000)]))  # 0.5 s silence, 0.5 s loud
    rms = v.load_rms(tmp_path / "a.wav")
    assert len(rms) == 100
    assert rms[10] == pytest.approx(0.0)
    assert rms[90] == pytest.approx(10000, rel=1e-3)


def test_snap_moves_cut_edges_to_the_quietest_nearby_frame():
    rms = np.ones(1000)  # 10 s
    rms[105] = 0  # dip just after the keep-start
    rms[395] = 0  # dip just before the keep-end
    assert v.snap([[1.0, 4.0, 1]], rms, 0.08) == [[pytest.approx(1.05), pytest.approx(3.95), 1]]


def test_snap_leaves_file_edges_alone_and_can_be_disabled():
    rms = np.ones(1000)
    keep = [[0.0, 10.0, 1]]
    assert v.snap(keep, rms, 0.08) == [[0.0, 10.0, 1]]
    assert v.snap(keep, rms, 0) is keep


def test_snap_drops_segments_that_collapse():
    rms = np.ones(1000)
    rms[100] = 0  # both edges of the 1.00-1.04 s segment snap to this one dip
    assert v.snap([[1.0, 1.04, 1]], rms, 0.08) == []


# --- vocabulary --------------------------------------------------------------------------------


def test_apply_vocab_with_an_empty_vocab_is_a_no_op():
    ws = spoken("hello", "world.")
    assert v.apply_vocab(ws, {}) == []
    assert [w["w"] for w in ws] == ["hello", "world."]


# --- captions ----------------------------------------------------------------------------------


def test_write_srt_applies_replacements_and_merges_extra_captions(tmp_path):
    out = tmp_path / "o.srt"
    v.write_srt(spoken("teh", "plan."), [[0.0, 2.0, 1]], out, {"teh": "the"}, extra=[(5.0, 6.0, "music")])
    first, second = out.read_text().strip().split("\n\n")
    assert first.splitlines()[2] == "the plan."
    assert second.splitlines()[0] == "2" and second.splitlines()[2] == "music"


def test_write_srt_wraps_long_runs_at_maxw(tmp_path):
    out = tmp_path / "o.srt"
    v.write_srt(spoken(*[f"w{i}" for i in range(10)]), [[0.0, 10.0, 1]], out, maxw=4)
    assert out.read_text().count("-->") == 3  # 4 + 4 + 2 words


# --- analysis-only CLI run ---------------------------------------------------------------------


@pytest.fixture
def stub_media(monkeypatch):
    """Stub ffprobe, ffmpeg silence detection and the audio RMS so main() runs without media."""
    monkeypatch.setattr(v, "LEAD", 0.0)  # main() reassigns these module globals
    monkeypatch.setattr(v, "PHRASES", list(v.PHRASES))
    monkeypatch.setattr(v, "duration", lambda p: 20.0)
    monkeypatch.setattr(v, "sh", lambda cmd: "30000/1001\n")  # ffprobe r_frame_rate
    monkeypatch.setattr(v, "silences", lambda *a, **k: [])
    monkeypatch.setattr(v, "load_rms", lambda wav: np.ones(2000))


def run_cli(monkeypatch, tmp_path, *extra):
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["vidtight", str(tmp_path / "clip.mov"), "--out", str(out), *extra])
    v.main()
    return out


@pytest.fixture
def vocab_file(tmp_path):
    f = tmp_path / "vocab.yaml"
    f.write_text("GravityKit: [Gravity Kit]\n")
    return f


def test_main_writes_edl_cuts_csv_and_srt(stub_media, monkeypatch, tmp_path, vocab_file, capsys):
    monkeypatch.setattr(v, "transcribe", lambda src, work, model: (spoken("So", "um,", "Gravity", "Kit", "is", "great.", "the", "end."), 2.0))
    out = run_cli(monkeypatch, tmp_path, "--vocab", str(vocab_file))

    edl = json.loads((out / "edl.json").read_text())
    assert {c[2] for c in edl["cuts"]} == {"filler"}
    assert [f["to"] for f in edl["vocab_fixes"]] == ["GravityKit"]
    assert edl["keep"] and all(s < e for s, e, _ in edl["keep"])

    rows = (out / "cuts.csv").read_text().splitlines()
    assert rows[0] == "start,end,type,detail" and len(rows) == 3  # header + "So" + "um,"

    srt = (out / "clip.srt").read_text()
    assert "GravityKit is great." in srt and "um" not in srt

    assert (out / "words.json").exists()
    stdout = capsys.readouterr().out
    assert "cuts: {'filler': 2}" in stdout and "vocab fixes: 1" in stdout and f"outputs in {out}" in stdout


def test_main_reuse_skips_asr(stub_media, monkeypatch, tmp_path, vocab_file):
    out = tmp_path / "out"
    out.mkdir()
    (out / "words.json").write_text(json.dumps(spoken("Hello", "world.")))

    def no_asr(*a, **k):
        raise AssertionError("--reuse must not re-run ASR")

    monkeypatch.setattr(v, "transcribe", no_asr)
    run_cli(monkeypatch, tmp_path, "--vocab", str(vocab_file), "--reuse")
    assert "Hello world." in (out / "clip.srt").read_text()


def test_main_applies_manual_cut_and_caption_replacements(stub_media, monkeypatch, tmp_path, vocab_file):
    monkeypatch.setattr(v, "transcribe", lambda src, work, model: (spoken("teh", "plan.", "tangent", "words", "here."), 1.0))
    edits = tmp_path / "edits.yaml"
    edits.write_text('cut:\n  - [1.0, 2.0, "tangent"]\nreplace:\n  "teh": "the"\n')
    out = run_cli(monkeypatch, tmp_path, "--vocab", str(vocab_file), "--edits", str(edits))

    srt = (out / "clip.srt").read_text()
    assert "the plan." in srt and "here." in srt
    assert "tangent" not in srt
    assert 'manual,"tangent"' in (out / "cuts.csv").read_text()


def test_main_turns_a_listed_pause_into_a_speed_up(stub_media, monkeypatch, tmp_path, vocab_file):
    monkeypatch.setattr(v, "silences", lambda wav, noise, mind: [(2.0, 5.0)] if mind == 0.8 else [])
    monkeypatch.setattr(v, "transcribe", lambda src, work, model: (spoken("one", "two", "three", "four."), 1.0))
    edits = tmp_path / "edits.yaml"
    edits.write_text("speed:\n  - [3.0, 4]\n")
    out = run_cli(monkeypatch, tmp_path, "--vocab", str(vocab_file), "--edits", str(edits))

    edl = json.loads((out / "edl.json").read_text())
    assert not [c for c in edl["cuts"] if c[2] == "pause"]  # the pause is sped up, not cut
    assert len(edl["speed"]) == 1 and edl["speed"][0][2] == 4
    assert any(sp == 4 for _, _, sp in edl["keep"])
    assert "speed x4" in (out / "cuts.csv").read_text()


def test_main_runs_without_a_bundled_vocab(stub_media, monkeypatch, tmp_path):
    # A non-editable install has no vocab.yaml next to the module; the default must not crash.
    monkeypatch.setattr(v, "__file__", str(tmp_path / "site-packages" / "vidtight.py"))
    monkeypatch.setattr(v, "transcribe", lambda src, work, model: (spoken("Hello", "world."), 1.0))
    out = run_cli(monkeypatch, tmp_path)
    assert json.loads((out / "edl.json").read_text())["vocab_fixes"] == []


def test_main_still_fails_on_an_explicit_missing_vocab(stub_media, monkeypatch, tmp_path):
    monkeypatch.setattr(v, "transcribe", lambda src, work, model: (spoken("Hello", "world."), 1.0))
    with pytest.raises(FileNotFoundError):
        run_cli(monkeypatch, tmp_path, "--vocab", str(tmp_path / "nope.yaml"))


# --- true peak, contact sheet ------------------------------------------------------------------

EBUR = """
[Parsed_ebur128_0] t: 0.1  I: -70.0 LUFS  TPK: -9.9 dBFS
[Parsed_ebur128_0] Summary:

  Integrated loudness:
    I:         -12.2 LUFS
    Threshold: -22.4 LUFS

  True peak:
    Peak:       -1.1 dBFS
"""

LOUDNORM = """
[Parsed_loudnorm_0]
{
	"input_i" : "-12.20",
	"input_tp" : "-0.50",
	"input_lra" : "4.20",
	"input_thresh" : "-22.50",
	"output_i" : "-12.50",
	"output_tp" : "-1.50",
	"output_lra" : "4.20",
	"output_thresh" : "-22.80",
	"normalization_type" : "dynamic",
	"target_offset" : "0.00"
}
"""


def test_parse_loudness_prefers_the_ebur128_summary_over_frame_lines():
    # The per-frame log also contains "I:" and "dBFS". Only the summary counts, and it is the encoded file's true peak.
    assert v.parse_loudness(EBUR) == (-1.1, -12.2)
    assert v.parse_loudness(LOUDNORM) == (-0.5, -12.2)
    assert v.parse_loudness("no meter here") is None


def test_measure_loudness_reads_ebur128_and_falls_back_to_loudnorm(monkeypatch, tmp_path):
    calls = []

    def meter(cmd):
        return any("ebur128" in str(part) for part in cmd)

    def fake_run(cmd, **kw):
        calls.append(cmd)
        err = EBUR if len(calls) == 1 else LOUDNORM
        return SimpleNamespace(returncode=0, stdout="", stderr=err if meter(cmd) else "")

    monkeypatch.setattr(v.subprocess, "run", fake_run)
    target = tmp_path / "out.m4a"
    assert v.measure_loudness(target) == (-1.1, -12.2)
    assert Path(calls[0][calls[0].index("-i") + 1]).is_absolute()
    assert "ebur128=peak=true:framelog=quiet" in calls[0]
    assert len(calls) == 1  # the summary parsed, so loudnorm was not needed

    calls.clear()
    monkeypatch.setattr(v.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="", stderr="" if meter(cmd) else LOUDNORM))
    assert v.measure_loudness(target) == (-0.5, -12.2)


def test_true_peak_guard_reencodes_audio_until_the_encoded_peak_clears(capsys):
    limits = []

    def encode(limit):
        limits.append(limit)

    peaks = iter([(-0.5, -12.2), (-1.1, -12.2)])
    limit, tp, lufs = v.guard_true_peak(encode, lambda: next(peaks))
    assert limits[0] == pytest.approx(0.83) and limits[1] < 0.83
    assert "aresample=192000" in v.limiter_chain(limits[0]) and f"alimiter=limit={limits[0]:.4f}" in v.limiter_chain(limits[0])
    assert (limit, tp, lufs) == (limits[-1], -1.1, -12.2)
    out = capsys.readouterr().out
    assert "re-encoding audio" in out and "loudness: -12.2 LUFS, -1.1 dBTP" in out


def test_true_peak_guard_stops_when_the_encoded_peak_already_passes(capsys):
    limits = []
    v.guard_true_peak(lambda limit: limits.append(limit), lambda: (-1.4, -12.2))
    assert limits == [pytest.approx(0.83)]
    assert "re-encoding" not in capsys.readouterr().out


def test_true_peak_guard_warns_when_the_peak_never_clears(capsys):
    v.guard_true_peak(lambda limit: None, lambda: (0.2, -12.0), tries=2)
    err = capsys.readouterr().err
    assert "WARNING" in err and "0.2 dBTP" in err


def test_render_audio_measures_the_encoded_file_and_lowers_the_limiter(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(v, "LEAD", 0.0)
    out = tmp_path / "clip.audio.m4a"
    raw = out.with_suffix(".src.s16")
    raw.write_bytes(np.zeros((4800, 2), np.int16).tobytes())  # 0.1 s of stereo 48 kHz, so the extract is skipped
    cmds = []

    def fake_sh(cmd):
        cmds.append(cmd)
        Path(cmd[-1]).write_bytes(b"m4a")

    peaks = iter([(-0.5, -12.2), (-1.1, -12.2)])
    monkeypatch.setattr(v, "sh", fake_sh)
    monkeypatch.setattr(v, "measure_loudness", lambda path: next(peaks))
    v.render_audio(tmp_path / "clip.mov", [[0.0, 0.1, 1]], out)
    encodes = [c for c in cmds if "aac_at" in c]
    assert len(encodes) == 2  # the hot encode, then the audio only, at a lower ceiling
    graphs = [c[c.index("-filter_complex") + 1] for c in encodes]
    assert "aresample=192000" in graphs[0] and "alimiter=limit=0.8300" in graphs[0]
    assert "alimiter=limit=0.8300" not in graphs[1]
    limit = float(graphs[1].split("alimiter=limit=")[1].split(":")[0])
    assert limit < 0.83
    assert "loudness: -12.2 LUFS, -1.1 dBTP" in capsys.readouterr().out


def test_contact_times_are_every_n_seconds_and_include_the_start():
    assert v.contact_times(24, 10) == [0, 10, 20]
    assert v.contact_times(10, 5) == [0, 5]
    assert v.contact_times(0.4, 10) == [0.0]
    with pytest.raises(ValueError):
        v.contact_times(10, 0)


def test_tile_contact_normalises_mixed_frame_sizes_and_draws_a_label(tmp_path):
    from PIL import Image

    wide, tall = tmp_path / "wide.png", tmp_path / "tall.png"
    Image.new("RGB", (100, 40), (200, 0, 0)).save(wide)
    Image.new("RGB", (40, 100), (0, 0, 200)).save(tall)
    dest = tmp_path / "sheet" / "contact.png"
    v.tile_contact([(wide, "00:00:00.000"), (tall, "00:00:10.000")], dest, cell_w=40)
    sheet = Image.open(dest)
    assert sheet.size == (80, 100 + 24)  # two cells, width-normalised, padded to the taller frame, plus the label bar
    assert sheet.getpixel((20, 50)) == (200, 0, 0)  # the wide frame sits in the middle of its cell, not stretched
    assert sheet.getpixel((0, 0)) == (0, 0, 0)  # the pad above it
    assert sheet.getpixel((40 + 20, 50)) == (0, 0, 200)
    bar = [sheet.getpixel((x, 100 + 8)) for x in range(6, 36)]
    assert any(px[0] > 80 for px in bar)  # the timecode is drawn into the bar, not left as the bar colour


def test_contact_sheet_extracts_frames_by_absolute_path(monkeypatch, tmp_path, capsys):
    cmds = []
    monkeypatch.setattr(v, "duration", lambda p: 10.0)
    monkeypatch.setattr(v, "sh", lambda cmd: cmds.append(list(cmd)))

    def fake_tile(frames, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"png")
        assert [label for _, label in frames] == ["00:00:00.000", "00:00:05.000"]

    monkeypatch.setattr(v, "tile_contact", fake_tile)
    src = tmp_path / "clip.mov"
    src.write_bytes(b"")
    dest = tmp_path / "out" / "contact.png"
    v.contact_sheet(src, dest, 5)
    assert len(cmds) == 2  # t = 0 and t = 5
    for cmd in cmds:
        assert Path(cmd[cmd.index("-i") + 1]) == src.resolve()
        assert Path(cmd[-1]).is_absolute()
    text = capsys.readouterr().out
    assert "contact sheet:" in text and "every 5s" in text


def test_main_contact_flag_writes_the_sheet_beside_the_analysis(stub_media, monkeypatch, tmp_path):
    seen = {}

    def fake(src, dest, every):
        seen.update(src=Path(src), dest=Path(dest), every=every)
        Path(dest).write_bytes(b"png")

    monkeypatch.setattr(v, "contact_sheet", fake)
    monkeypatch.setattr(v, "transcribe", lambda src, work, model: (spoken("Hello", "world."), 1.0))
    out = run_cli(monkeypatch, tmp_path, "--contact", "4")
    assert seen["every"] == 4 and seen["dest"] == out / "contact.png" and seen["src"].name == "clip.mov"
    assert (out / "contact.png").exists() and (out / "edl.json").exists()


def test_main_rejects_a_non_positive_contact_interval(stub_media, monkeypatch, tmp_path):
    monkeypatch.setattr(v, "transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should fail before ASR")))
    monkeypatch.setattr(sys, "argv", ["vidtight", "clip.mov", "--out", str(tmp_path), "--contact", "0"])
    with pytest.raises(SystemExit):
        v.main()
