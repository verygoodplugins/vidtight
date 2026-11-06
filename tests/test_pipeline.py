"""Tests for the I/O edges of vidtight.py: ffmpeg wrappers, ASR token merging, RMS loading and the analysis-only CLI run.

ffmpeg, ffprobe and parakeet-mlx are all stubbed, so these run on any machine without media or Apple Silicon.
"""
import json
import sys
import wave
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
