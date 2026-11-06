"""Unit tests for the pure pipeline logic in vidtight.py (no ASR, no ffmpeg, no media)."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vidtight as v

ROOT = Path(__file__).resolve().parent.parent


def words(*items):
    """Build word dicts spaced 0.5s apart; a trailing '.' marks end of sentence."""
    out = []
    for i, w in enumerate(items):
        d = {"w": w, "s": i * 0.5, "e": i * 0.5 + 0.4}
        if w.endswith("."):
            d["eos"] = True
        out.append(d)
    return out


def args(**kw):
    base = dict(wav="unused.wav", dur=20.0, noise=-35, retake_words=5, pause=0.8, pad=0.15, mincut=0.25)
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture
def no_silence(monkeypatch):
    monkeypatch.setattr(v, "silences", lambda *a, **k: [])


def test_norm_strips_punctuation_but_keeps_apostrophes():
    assert v.norm("Hello,") == "hello"
    assert v.norm("Don't!") == "don't"


def test_build_edl_inverts_cuts_into_keep_segments():
    assert v.build_edl([(1, 2, "pause", ""), (5, 6, "filler", "um")], 10) == [[0, 1, 1], [2, 5, 1], [6, 10, 1]]


def test_build_edl_merges_overlapping_cuts_and_drops_slivers():
    keep = v.build_edl([(0, 2, "a", ""), (1.9, 4, "b", ""), (4.02, 9.99, "c", "")], 10)
    assert keep == []  # 0-4 merged, 4.02-9.99 cut, slivers under 50 ms dropped


def test_remap_maps_source_time_to_edited_time():
    keep = [[0, 1, 1], [3, 5, 1]]
    assert v.remap(0.5, keep) == pytest.approx(0.5)
    assert v.remap(2.0, keep) == pytest.approx(1.0)  # inside the cut: snaps to the join
    assert v.remap(4.0, keep) == pytest.approx(2.0)


def test_apply_speed_splits_a_keep_segment():
    assert v.apply_speed([[0, 10, 1]], [(2, 4, 4)]) == [[0, 2, 1], [2, 4, 4], [4, 10, 1]]
    assert v.seglen([2, 4, 4]) == pytest.approx(0.5)


def test_quantize_snaps_to_frame_boundaries():
    (s, e, sp), = v.quantize([[0.01, 1.02, 1]], 30)
    assert (s, sp) == (0.0, 1)
    assert e == pytest.approx(31 / 30)


def test_timestamp_formats():
    assert v.ts(61.25) == "00:01:01,250"
    assert v.ts(3661.5, ".") == "01:01:01.500"


def test_quiet_finds_the_quietest_frame_in_range():
    rms = np.ones(1000)
    rms[250] = 0
    assert v.quiet(rms, 2.4, 2.6) == pytest.approx(2.5)


def test_fix_fillers_moves_start_back_to_the_preceding_dip():
    rms = np.ones(1000)
    rms[100] = 0  # silence at 1.00 s, just before the filler
    (cut,) = v.fix_fillers([(1.2, 1.5, "filler", "um")], rms)
    assert cut[0] == pytest.approx(1.0)
    assert cut[1:] == (1.5, "filler", "um")
    assert v.fix_fillers([(1.2, 1.5, "pause", "2s")], rms) == [(1.2, 1.5, "pause", "2s")]  # only fillers move


def test_apply_vocab_explicit_variant_keeps_punctuation():
    ws = words("Gravity", "Kit,")
    fixes = v.apply_vocab(ws, {"GravityKit": ["Gravity Kit"]})
    assert [w["w"] for w in ws] == ["GravityKit,"]
    assert ws[0]["e"] == pytest.approx(0.9)
    assert fixes == [{"t": 0.0, "from": "Gravity Kit,", "to": "GravityKit,"}]


def test_apply_vocab_fuzzy_matches_near_misses():
    ws = words("Freemeus")
    fixes = v.apply_vocab(ws, {"Freemius": []})
    assert ws[0]["w"] == "Freemius"
    assert fixes[0]["from"] == "Freemeus"


def test_apply_vocab_leaves_unrelated_words_alone():
    ws = words("hello", "world.")
    assert v.apply_vocab(ws, {"WP Fusion": ["WP fusion"]}) == []
    assert [w["w"] for w in ws] == ["hello", "world."]


def test_find_cuts_flags_fillers_and_sentence_start_softeners(no_silence):
    ws = words("So", "um,", "this", "is", "great.")
    cuts = v.find_cuts(ws, args())
    assert [(c[2], c[3]) for c in cuts] == [("filler", "So"), ("filler", "um,")]
    assert ws[0]["cut"] and ws[1]["cut"]


def test_find_cuts_flags_stutters(no_silence):
    ws = words("I", "think", "the", "the", "plan.")
    (cut,) = v.find_cuts(ws, args())
    assert cut[:3] == (1.0, 1.4, "stutter")


def test_find_cuts_drops_the_first_take_of_a_retake(no_silence):
    ws = words("we", "open", "the", "settings", "panel", "then", "click", "we", "open", "the", "settings", "panel", "and", "export.")
    (cut,) = v.find_cuts(ws, args())
    assert cut[2] == "retake"
    assert cut[0] == pytest.approx(0.0)
    assert cut[1] <= ws[7]["s"]  # ends before the second take starts
    assert all(w.get("cut") for w in ws[:7])


def test_find_cuts_turns_long_silences_into_pause_cuts(monkeypatch):
    monkeypatch.setattr(v, "silences", lambda wav, noise, mind: [(2.0, 5.0)] if mind == 0.8 else [])
    ws = words("one", "two", "three", "four.")
    (cut,) = v.find_cuts(ws, args())
    assert cut[:3] == (2.15, 4.85, "pause")  # padded by --pad on both sides


def test_write_srt_retimes_kept_words_and_skips_cut_ones(tmp_path):
    ws = words("Hello", "um", "world.")
    ws[1]["cut"] = True
    out = tmp_path / "out.srt"
    v.write_srt(ws, [[0.0, 2.0, 1]], out)
    text = out.read_text()
    assert "Hello world." in text
    assert "um" not in text
    assert text.startswith("1\n00:00:00,000 --> ")


def test_cli_help_lists_the_main_flags():
    out = subprocess.run([sys.executable, str(ROOT / "vidtight.py"), "--help"], capture_output=True, text=True, check=True).stdout
    for flag in ("--render", "--edits", "--vocab", "--retake-words", "--reuse"):
        assert flag in out
