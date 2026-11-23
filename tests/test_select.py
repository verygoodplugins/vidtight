"""Tests for moment selection (--select): the speech map, speech-safe edge placement and the EDL self-check.

Everything runs on synthetic word lists: no ASR, no ffmpeg, no media.
"""
import csv
import json
import random
import sys

import numpy as np
import pytest

import vidtight as v

DUR = 60.0


def say(text, start, wlen=0.3):
    """Words back to back from `start`, the way parakeet stamps a sentence: no gaps inside it."""
    out = []
    for i, w in enumerate(text.split()):
        d = {"w": w, "s": round(start + i * wlen, 3), "e": round(start + (i + 1) * wlen, 3)}
        if w[-1] in ".!?":
            d["eos"] = True
        out.append(d)
    return out


# 10.0-11.2 "Welcome to the show." | 0.8 s gap | 12.0-12.9 "Take a light." | 0.3 s gap | 13.2-14.1 "Raise it higher." | ... | 30.0-30.6 "The end."
WORDS = say("Welcome to the show.", 10.0) + say("Take a light.", 12.0) + say("Raise it higher.", 13.2) + say("The end.", 30.0)


def gaps_for(words, speech=(), **kw):
    blocks = v.speech_blocks(words, speech)
    return blocks, v.speech_gaps(blocks, DUR, **kw)


def pick(moments, words=WORDS, speech=(), **kw):
    blocks, gaps = gaps_for(words, speech)
    return v.select_moments(moments, gaps, blocks, DUR, **kw)


# --- speech map --------------------------------------------------------------------------------


def test_speech_blocks_cut_smeared_word_ends_back_to_a_plausible_length():
    # Music under the voice: parakeet stamped one word across 25 s. A five-letter word is capped at 0.3 + 5 * 0.1 s.
    words = say("was here all", 79.1) + [{"w": "along.", "s": 80.0, "e": 104.96, "eos": True}]
    blocks = v.speech_blocks(words)
    last = blocks[-1]
    assert (last["s"], last["src"], last["raw"]) == (80.0, "cap", 104.96)
    assert last["e"] == pytest.approx(80.8)
    assert [b["src"] for b in blocks[:-1]] == ["asr"] * 3  # ordinary words keep their stamps


def test_speech_blocks_never_let_a_word_run_past_max_word():
    (b,) = v.speech_blocks([{"w": "Supercalifragilisticexpialidocious", "s": 1.0, "e": 9.0}], maxword=1.2)
    assert b["e"] == pytest.approx(2.2)


def test_speech_blocks_trust_a_detected_silence_over_the_cap():
    words = [{"w": "along.", "s": 80.0, "e": 104.96, "eos": True}]
    (b,) = v.speech_blocks(words, sil=[(80.45, 100.0)])
    assert (b["e"], b["src"]) == (80.45, "silence")  # measured, not estimated


def test_caller_intervals_replace_the_asr_words_they_overlap():
    blocks = v.speech_blocks(WORDS, speech=[[11.9, 14.3, "tutorial line"]])
    texts = [b["w"] for b in blocks]
    assert "tutorial line" in texts
    assert not {"Take", "light.", "Raise", "higher."} & set(texts)  # overridden
    assert {"Welcome", "show.", "The", "end."} <= set(texts)  # untouched ASR words stay
    app = next(b for b in blocks if b["src"] == "app")
    assert (app["s"], app["e"]) == (11.9, 14.3)


def test_a_smeared_word_is_not_dropped_by_a_caller_interval_that_only_overlaps_its_smear():
    words = [{"w": "along.", "s": 80.0, "e": 104.96, "eos": True}]
    blocks = v.speech_blocks(words, speech=[[90.0, 95.0, "later line"]])
    assert [b["w"] for b in blocks] == ["along.", "later line"]


def test_utterances_split_on_gaps_including_the_gap_a_cap_reveals():
    words = say("Poke it and it", 30.0) + [{"w": "pops.", "s": 31.2, "e": 36.0, "eos": True}] + say("Each one lights a gem.", 36.0)
    utts = v.utterances(v.speech_blocks(words), 0.4)
    assert [u["text"] for u in utts] == ["Poke it and it pops.", "Each one lights a gem."]
    assert utts[0]["e"] == pytest.approx(31.9)
    assert utts[0]["est"] == 1  # one word end in this utterance is an estimate


def test_gaps_are_graded_clean_tight_or_not_cuttable():
    words = say("Auto", 1.0) + say("Jack.", 1.46) + say("Hi there.", 2.0) + say("Bye.", 4.0)  # 0.16 s mid-name, 0.24 s after a sentence, 1.4 s
    _, gaps = gaps_for(words)
    grades = [(round(g["e"] - g["s"], 2), g["grade"]) for g in gaps]
    # ASR word times drift by more than a short gap is wide (measured: 0.39 s early to 0.17 s late), so ASR-only gaps under 0.4 s never count.
    assert grades == [(1.0, "clean"), (0.16, None), (0.24, None), (0.0, None), (1.4, "clean"), (55.7, "clean")]


def test_a_gap_under_the_preferred_width_counts_only_between_two_caller_intervals():
    _, gaps = gaps_for(say("Bye.", 4.0), speech=[[1.0, 1.76, "Auto Jack."], [2.0, 2.6, "Hi there."], [3.7, 3.8, "Oh."]])
    grades = [(round(g["e"] - g["s"], 2), g["grade"]) for g in gaps]
    assert grades == [(1.0, "clean"), (0.24, "tight"), (1.1, "clean"), (0.2, None), (55.7, "clean")]  # app|app is tight; app|ASR word is not


def test_gap_spans_keep_the_pads_and_split_narrow_gaps_pro_rata():
    _, gaps = gaps_for(WORDS)
    wide = next(g for g in gaps if g["s"] == 11.2)  # 0.8 s between "show." and "Take"
    assert (wide["lo"], wide["hi"]) == (pytest.approx(11.35), pytest.approx(11.8))  # 150 ms after a word, 200 ms before the next
    tight = next(g for g in gaps if g["s"] == 12.9)  # 0.3 s: narrower than both pads
    assert tight["lo"] == tight["hi"] == pytest.approx(12.9 + 0.3 * 0.15 / 0.35)
    first, last = gaps[0], gaps[-1]
    assert (first["lo"], first["hi"]) == (0.0, pytest.approx(9.8))  # the file start needs no pad
    assert (last["lo"], last["hi"]) == (pytest.approx(30.75), DUR)


# --- edge placement ----------------------------------------------------------------------------


def test_a_window_in_open_air_is_kept_as_requested():
    clips, (row,) = pick([{"t": 20.0, "label": "quiet bit"}])
    assert clips == [[19.0, 24.0]]
    assert (row["status"], row["clip"]) == ("kept", 1)


def test_a_window_that_cuts_an_utterance_is_widened_to_take_it_whole():
    clips, (row,) = pick([{"t": 10.5, "label": "title", "pre": 0.3, "post": 0.3}])  # 10.2-10.8: both edges inside words
    assert clips == [[pytest.approx(9.8), pytest.approx(11.35)]]  # 200 ms before "Welcome", 150 ms after "show."
    assert row["status"] == "widened"
    assert "Welcome" in row["note"] and "show." in row["note"]
    assert row["speech"] == "Welcome to the show."


def test_an_edge_moves_to_the_nearest_gap_when_that_is_closer_than_widening():
    clips, (row,) = pick([{"t": 11.6, "label": "after the line", "pre": 0.6, "post": 2.4}], words=say("Welcome to the show.", 10.0))
    assert clips == [[pytest.approx(11.35), 14.0]]  # in-edge was inside "show.": skip the line instead of taking 1.2 s more
    assert row["status"] == "trimmed"


def test_a_moment_is_dropped_when_the_whole_utterance_would_exceed_max_clip():
    clips, (row,) = pick([{"t": 10.5, "label": "title", "pre": 0.3, "post": 0.3}], maxclip=1.0)
    assert clips == []
    assert row["status"] == "dropped" and row["clip"] is None
    assert "--max-clip" in row["note"] and "Welcome" in row["note"]


def test_a_moment_is_dropped_rather_than_trimmed_below_half_its_window():
    # 20 s of unbroken speech with the moment in the middle: there is no gap to move to.
    words = say(" ".join(["word"] * 80), 5.0, wlen=0.25)
    clips, (row,) = pick([{"t": 15.0, "label": "mid-monologue"}], words=words)
    assert clips == [] and row["status"] == "dropped"


LINES = [[12.0, 12.9, "Take a light."], [13.2, 14.1, "Raise it higher."]]  # the same two lines as exact intervals from the app
RUN_ON = WORDS[:4] + say("and then some more words follow on.", 14.1)  # speech carries straight on after "higher."


def test_tight_gaps_are_taken_only_when_they_save_more_than_a_second_of_movement():
    # The out-edge is wanted at 13.6, inside "Raise it higher." The tight gap (0.3 s, between the app's two lines) offers 13.03: 0.57 s away.
    moment = [{"t": 12.1, "label": "raise", "pre": 0.5, "post": 1.5}]
    clean_close, _ = pick(moment, speech=LINES)  # the clean gap after "higher." is 0.65 s away: not worth a tight gap
    clean_far, _ = pick(moment, words=RUN_ON, speech=LINES)  # the clean gap is now 2.75 s away
    assert clean_close == [[11.6, pytest.approx(14.25)]]
    assert clean_far == [[11.6, pytest.approx(12.9 + 0.3 * 0.15 / 0.35)]]


def test_a_short_gap_seen_only_by_the_asr_is_never_used():
    # Same timings, but from ASR words alone: the 0.3 s gap after "light." may not be where the ASR says it is.
    clips, _ = pick([{"t": 12.1, "label": "raise", "pre": 0.5, "post": 1.5}], words=WORDS[:10] + RUN_ON[4:])
    assert clips == [[11.6, pytest.approx(16.35)]]  # runs to the clean gap after "on." instead


def test_a_mid_sentence_gap_is_never_used_even_when_it_is_the_nearest():
    words = say("Auto", 10.0) + say("Jack is here.", 10.46)  # 0.16 s gap inside the name
    clips, _ = pick([{"t": 10.1, "label": "name", "pre": 0.05, "post": 0.3}], words=words)
    assert clips == [[pytest.approx(9.8), pytest.approx(11.51)]]  # whole line, not "Auto |"


def test_caller_intervals_are_never_cut_and_beat_the_asr_words_under_them():
    clips, (row,) = pick([{"t": 12.5, "label": "tutorial", "pre": 0.2, "post": 0.4}], speech=[[11.9, 14.3, "tutorial line"]])
    assert clips == [[pytest.approx(11.7), pytest.approx(14.45)]]
    assert row["speech"] == "tutorial line"


def test_overlapping_moments_merge_into_one_clip():
    clips, rows = pick([{"t": 20.0, "label": "a"}, {"t": 23.0, "label": "b"}, {"t": 40.0, "label": "c"}])
    assert clips == [[19.0, 27.0], [39.0, 44.0]]
    assert [r["clip"] for r in rows] == [1, 1, 2]


def test_clips_come_out_in_source_order_and_rows_keep_the_callers_order():
    clips, rows = pick([{"t": 40.0, "label": "late"}, {"t": 20.0, "label": "early"}])
    assert clips == [[19.0, 24.0], [39.0, 44.0]]
    assert [(r["label"], r["clip"]) for r in rows] == [("late", 2), ("early", 1)]


def test_per_moment_pre_and_post_override_the_defaults_and_bad_times_are_dropped():
    clips, rows = pick([{"t": 20.0, "label": "long", "pre": 2, "post": 6}, {"t": 99.0, "label": "past the end"}, {"t": 0.5, "label": "at the start"}], pre=1.0, post=4.0)
    assert clips == [[0.0, 4.5], [18.0, 26.0]]
    assert [r["status"] for r in rows] == ["kept", "dropped", "kept"]
    assert "outside" in rows[1]["note"]


# --- air ---------------------------------------------------------------------------------------

PAUSED = [[10.0, 12.0, "one"], [15.0, 17.0, "two"], [17.9, 19.0, "three"]]  # a 3 s pause, then a 0.9 s pause


def test_air_keeps_voice_free_footage_at_each_edge_where_the_pause_allows():
    gaps = v.speech_gaps(v.speech_blocks([], PAUSED), DUR, air=1.0)
    spans = [(round(g["lo"], 3), round(g["hi"], 3)) for g in gaps]
    # a second clear of speech where the pause has room for it on both sides, the middle of the pause where it does not
    assert spans == [(0.0, 9.0), (13.0, 14.0), (17.45, 17.45), (20.0, DUR)]
    assert [g["grade"] for g in gaps] == ["clean"] * 4


def test_air_turns_tight_gaps_off():
    lines = [[10.0, 12.0, "one"], [12.3, 14.0, "two"]]
    grade = lambda **kw: v.speech_gaps(v.speech_blocks([], lines), DUR, **kw)[1]["grade"]  # noqa: E731
    assert grade() == "tight" and grade(air=1.0) is None  # a 0.3 s gap cannot give any air


def test_with_air_a_clip_starts_and_ends_in_voice_free_footage():
    blocks = v.speech_blocks([], PAUSED)
    moment = [{"t": 11.0, "label": "mid-line", "pre": 0.5, "post": 0.5}]
    tight, _ = v.select_moments(moment, v.speech_gaps(blocks, DUR), blocks, DUR)
    aired, _ = v.select_moments(moment, v.speech_gaps(blocks, DUR, air=1.0), blocks, DUR)
    assert tight == [[pytest.approx(9.8), pytest.approx(12.15)]]  # right on the pads: the line and nothing else
    assert aired == [[9.0, 13.0]]


def test_clips_closer_than_the_join_distance_keep_the_pause_between_them():
    blocks = v.speech_blocks([], PAUSED)
    gaps = v.speech_gaps(blocks, DUR, air=1.0)
    moments = [{"t": 11.0, "label": "a", "pre": 0.5, "post": 0.5}, {"t": 16.0, "label": "b", "pre": 0.5, "post": 0.5}]
    apart, _ = v.select_moments(moments, gaps, blocks, DUR)
    joined, rows = v.select_moments(moments, gaps, blocks, DUR, join=2.0)
    assert apart == [[9.0, 13.0], [14.0, 17.45]]  # a 1 s jump inside one pause
    assert joined == [[9.0, 17.45]] and [r["clip"] for r in rows] == [1, 1]  # the whole pause stays in instead


# --- self-check --------------------------------------------------------------------------------


def test_check_edl_counts_edges_inside_speech_and_explains_each_edge():
    blocks = v.speech_blocks(WORDS)
    inside, close, ok, _ = v.check_edl([[10.45, 11.21, 1], [20.0, 25.0, 1]], blocks, DUR)
    start, end = v.check_edl([[0.0, DUR, 1]], blocks, DUR)
    assert not inside["safe"] and inside["inside"] and 'inside "to"' in inside["why"]
    assert not close["safe"] and not close["inside"]  # 10 ms after "show.": not in a word, but under the 30 ms minimum
    assert ok["safe"] and "after \"higher.\"" in ok["why"] and "before \"The\"" in ok["why"]
    assert start["safe"] and start["why"] == "file start" and end["safe"] and end["why"] == "file end"


def test_check_edl_exempts_only_the_exact_file_boundaries():
    blocks = v.speech_blocks([], [[9, 10, "last line"]])
    (_, short), (_, exact) = v.check_edl([[8, 9.96, 1]], blocks, 10), v.check_edl([[8, 10, 1]], blocks, 10)
    assert not short["safe"] and short["inside"] and 'inside "last line"' in short["why"]  # 40 ms short of the end still cuts the line
    assert exact["safe"] and exact["why"] == "file end"


def test_a_clip_that_runs_to_the_end_of_the_file_is_not_rounded_back_into_the_last_word():
    blocks = v.speech_blocks([], [[9.5, 10.01, "last line"]])  # speech runs to the end of a 10.01 s file: 10.01 is not on the 30 fps grid
    gaps = v.speech_gaps(blocks, 10.01)
    clips, _ = v.select_moments([{"t": 9.7, "label": "ending", "pre": 0.3, "post": 0.3}], gaps, blocks, 10.01)
    keep = v.settle(clips, gaps, np.ones(1001), 0.08, 30, 10.01)
    (s, e, _), = keep
    assert 9.2 <= s <= 9.3 and e == 10.01  # plain rounding would put the out-edge at 10.0, inside the line
    assert all(e["safe"] for e in v.check_edl(keep, blocks, 10.01))


def test_check_edl_marks_edges_that_rest_on_an_estimated_word_end():
    words = say("it", 30.9) + [{"w": "pops.", "s": 31.2, "e": 36.0, "eos": True}] + say("Each one.", 42.8)
    blocks = v.speech_blocks(words)
    a, b = v.check_edl([[33.9, 38.9, 1]], blocks, DUR)
    assert a["safe"] and a["est"]  # safe only if "pops." really ended by 31.9; ASR said 36.0
    assert "estimate" in a["why"]
    assert b["safe"] and not b["est"]  # 38.9 is past even the smeared stamp


def test_snap_with_a_limit_cannot_eat_into_the_pads():
    rms = np.ones(6000)
    rms[985] = 0  # quietest frame near the in-edge: 150 ms before "Welcome", inside its 200 ms lead
    rms[1128] = 0  # quietest frame near the out-edge: 80 ms after "show.", inside its 150 ms tail
    _, gaps = gaps_for(WORDS)
    lim = lambda t: next((g["lo"], g["hi"]) for g in gaps if g["s"] <= t <= g["e"])  # noqa: E731
    assert v.snap([[9.8, 11.35, 1]], rms, 0.08) == [[pytest.approx(9.85), pytest.approx(11.28), 1]]  # plain snap chases the dips toward the words
    (s, e, _), = v.snap([[9.8, 11.35, 1]], rms, 0.08, lim)
    assert 9.77 <= s <= 9.8 and e == pytest.approx(11.35)  # fenced: the edge may only move away from speech
    pinned = 12.9 + 0.3 * 0.15 / 0.35  # a gap narrower than both pads offers a single point
    assert v.snap([[pinned, 20.0, 1]], rms, 0.08, lim)[0][0] == pytest.approx(pinned)


def test_selection_never_leaves_an_edge_inside_speech_on_random_transcripts():
    rng = random.Random(7)
    for _ in range(150):
        t, words = rng.uniform(0, 3), []
        for _ in range(rng.randint(5, 60)):
            w = {"w": rng.choice(["a", "light", "column.", "cathedral", "pops.", "and", "together.", "Jack!"]), "s": round(t, 3)}
            t += rng.uniform(0.08, 0.5)
            w["e"] = round(t + (rng.choice([0, 0, 0, 3.0, 20.0]) if w["w"][-1] in ".!" else 0), 3)  # some sentence ends smear
            t = max(t, w["s"] + 0.01) + rng.choice([0, 0, 0, 0.08, 0.16, 0.3, 0.5, 2.0, 6.0])
            words.append(w)
        dur = t + rng.uniform(0, 5)
        fps = rng.choice([24, 30, 60])
        blocks = v.speech_blocks(words)
        gaps = v.speech_gaps(blocks, dur, tail=0.15 + 0.5 / fps, floor=v.MINPAD + 0.5 / fps)
        moments = [{"t": rng.uniform(0, dur), "label": str(i)} for i in range(rng.randint(1, 8))]
        clips, rows = v.select_moments(moments, gaps, blocks, dur)
        rms = np.array([rng.random() for _ in range(int(dur * 100) + 1)])
        keep = v.settle(clips, gaps, rms, 0.08, fps, dur)
        bad = [e for e in v.check_edl(keep, blocks, dur) if not e["safe"]]
        assert not bad, (words, moments, bad)
        assert len(keep) == len(clips)
        for r in rows:
            if r["clip"]:
                s, e, _ = keep[r["clip"] - 1]
                assert s - 0.15 <= r["t"] <= e + 0.15  # the moment itself is still in its clip (snap and frame rounding move an edge by up to 0.1 s)


def test_selection_between_exact_app_lines_stays_speech_safe_on_random_timelines():
    rng = random.Random(11)
    for _ in range(100):
        t, speech = rng.uniform(0, 3), []
        for i in range(rng.randint(2, 25)):
            e = t + rng.uniform(0.4, 4.0)
            speech.append([round(t, 3), round(e, 3), f"line {i}"])
            t = e + rng.choice([0.05, 0.16, 0.2, 0.3, 0.38, 0.5, 2.0, 6.0])
        dur, fps = rng.choice([speech[-1][1], speech[-1][1] + 0.02, t + rng.uniform(0, 5)]), rng.choice([24, 30, 60])  # sometimes the last line runs to the end
        air = rng.choice([0, 0, 0.5, 1.0, 2.5])
        blocks = v.speech_blocks([], speech)
        gaps = v.speech_gaps(blocks, dur, tail=0.15 + 0.5 / fps, floor=v.MINPAD + 0.5 / fps, air=air)
        clips, _ = v.select_moments([{"t": rng.uniform(0, dur), "label": str(i)} for i in range(rng.randint(1, 8))], gaps, blocks, dur, join=max(v.JOIN, 2 * air))
        keep = v.settle(clips, gaps, np.ones(int(dur * 100) + 1), 0.08, fps, dur)
        bad = [e for e in v.check_edl(keep, blocks, dur) if not e["safe"]]
        assert not bad, (speech, bad)


# --- frame rate --------------------------------------------------------------------------------


def test_probe_fps_falls_back_to_the_average_rate_for_tick_rate_containers(monkeypatch):
    monkeypatch.setattr(v, "sh", lambda cmd: "100000/1,482150000/8036473\n")  # Quest capture: r_frame_rate is the timebase
    assert v.probe_fps("quest.mp4") == 60
    monkeypatch.setattr(v, "sh", lambda cmd: "30000/1001,30000/1001\n")
    assert v.probe_fps("screen.mov") == 30


# --- CLI ---------------------------------------------------------------------------------------


@pytest.fixture
def stub_media(monkeypatch):
    monkeypatch.setattr(v, "LEAD", 0.0)
    monkeypatch.setattr(v, "PHRASES", list(v.PHRASES))
    monkeypatch.setattr(v, "duration", lambda p: DUR)
    monkeypatch.setattr(v, "sh", lambda cmd: "30/1,30/1\n")
    monkeypatch.setattr(v, "silences", lambda *a, **k: [])
    monkeypatch.setattr(v, "load_rms", lambda wav: np.ones(int(DUR * 100)))

    def no_asr(*a, **k):
        raise AssertionError("--words must not run ASR")

    monkeypatch.setattr(v, "transcribe", no_asr)


def run_select(monkeypatch, tmp_path, moments, *extra, words=WORDS):
    out = tmp_path / "out"
    (tmp_path / "words.json").write_text(json.dumps(words))
    (tmp_path / "moments.yaml").write_text(moments)
    argv = ["vidtight", str(tmp_path / "clip.mp4"), "--out", str(out), "--words", str(tmp_path / "words.json"), "--select", str(tmp_path / "moments.yaml"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    v.main()
    return out


MOMENTS = '- {t: 10.5, label: "title", pre: 0.3, post: 0.3, caption: "The show"}\n- {t: 20.0, label: "quiet bit"}\n- {t: 99.0, label: "nope"}\n'


def test_main_select_writes_an_edl_in_the_existing_schema_and_a_select_csv(stub_media, monkeypatch, tmp_path, capsys):
    out = run_select(monkeypatch, tmp_path, MOMENTS, "--snap", "0")

    edl = json.loads((out / "edl.json").read_text())
    assert {"src", "keep", "cuts", "speed", "vocab_fixes"} <= set(edl)  # same keys the tighten mode writes
    assert edl["keep"] == [[pytest.approx(9.8), pytest.approx(11.3667, abs=1e-3), 1], [19.0, 24.0, 1]]  # out-edge: 150 ms + half a frame after "show.", on the 30 fps grid
    assert [c[2] for c in edl["cuts"]] == ["unselected"] * 3 and edl["speed"] == []
    assert [m["status"] for m in edl["select"]] == ["widened", "kept", "dropped"]

    rows = list(csv.DictReader((out / "select.csv").open()))
    assert [r["moment"] for r in rows] == ["title", "quiet bit", "nope"]
    assert (rows[0]["in_safe"], rows[0]["out_safe"]) == ("yes", "yes")
    assert 'before "Welcome"' in rows[0]["in_why"] and 'after "show."' in rows[0]["out_why"]
    assert float(rows[0]["in"]) == pytest.approx(9.8) and rows[0]["speech"] == "Welcome to the show."
    assert rows[2]["status"] == "dropped" and rows[2]["in"] == ""

    srt = (out / "clip.srt").read_text()
    assert "Welcome to the show." in srt and "The show" in srt  # spoken words and the moment caption, retimed
    assert "Take a light." not in srt  # not selected

    stdout = capsys.readouterr().out
    assert "self-check: 4 edges re-read from edl.json: 0 inside speech" in stdout
    assert "LOW CONFIDENCE" not in stdout
    assert (out / "words.json").exists()


def test_main_select_maps_speech_from_the_raw_words_not_the_vocabulary_merged_ones(stub_media, monkeypatch, tmp_path):
    # The vocabulary fix merges three 0.6 s words into one 1.8 s span. Capped as a single word it would end at 11.2, mid-"Plugin", and the
    # gap that opened there would take the out-edge. The speech map has to see the three words as transcribed.
    words = say("Very Good Plugin today.", 10.0, wlen=0.6)
    (tmp_path / "vocab.yaml").write_text("Very Good Plugins: [Very Good Plugin]\n")
    out = run_select(monkeypatch, tmp_path, "- {t: 10.4, label: brand, pre: 0.4, post: 0.8}\n", "--vocab", str(tmp_path / "vocab.yaml"), "--snap", "0", words=words)
    edl = json.loads((out / "edl.json").read_text())
    (s, e, _), = edl["keep"]
    assert s == pytest.approx(9.8) and e == pytest.approx(12.4 + 0.15 + 1 / 60, abs=1e-3)  # the whole line, not 11.37 inside "Plugin"
    assert all(x["safe"] for x in v.check_edl(edl["keep"], v.speech_blocks(words), DUR))
    assert [f["to"] for f in edl["vocab_fixes"]] == ["Very Good Plugins"]
    assert "Very Good Plugins today." in (out / "clip.srt").read_text()  # captions still get the correction


def test_main_select_says_plainly_when_the_speech_map_is_low_confidence(stub_media, monkeypatch, tmp_path, capsys):
    words = say("Poke it and it", 30.0) + [{"w": "pops.", "s": 31.2, "e": 36.0, "eos": True}]
    run_select(monkeypatch, tmp_path, "- {t: 34.9, label: bells}\n", words=words)
    stdout = capsys.readouterr().out
    assert "LOW CONFIDENCE" in stdout and '"pops."' in stdout
    assert "1 of those words are stamped 3 s or longer" in stdout and "may be missing from the transcript" in stdout
    assert "1 rest on an estimated word end" in stdout


def test_main_select_reads_caller_speech_intervals(stub_media, monkeypatch, tmp_path):
    (tmp_path / "speech.yaml").write_text('- [18.5, 21.0, "app line"]\n')
    out = run_select(monkeypatch, tmp_path, "- {t: 20.0, label: quiet bit}\n", "--speech", str(tmp_path / "speech.yaml"), "--snap", "0")
    (s, e, _), = json.loads((out / "edl.json").read_text())["keep"]
    assert s == pytest.approx(18.3) and e == pytest.approx(24.0)  # in-edge moved out of the app's line


def test_main_select_air_leaves_room_around_the_line_and_joins_near_clips(stub_media, monkeypatch, tmp_path):
    (tmp_path / "speech.yaml").write_text('- [18.5, 21.0, "app line"]\n')
    moments = "- {t: 20.0, label: mid-line}\n- {t: 26.8, label: soon after}\n"
    args = ["--speech", str(tmp_path / "speech.yaml"), "--snap", "0"]
    keep = lambda *extra: json.loads((run_select(monkeypatch, tmp_path, moments, *args, *extra) / "edl.json").read_text())["keep"]  # noqa: E731
    assert keep() == [[pytest.approx(18.3), 24.0, 1], [25.8, 30.8, 1]]  # in-edge on its pad before the line, two clips
    assert keep("--air", "1") == [[17.5, pytest.approx(31.6), 1]]  # a second of air before the line and after "The end.", and the 1.8 s between the clips stays in


def test_main_select_warns_loudly_and_strict_exits_nonzero_when_an_edge_is_inside_speech(stub_media, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(v, "snap", lambda keep, rms, win, lim=None: [[10.45, e, sp] for s, e, sp in keep])  # sabotage: shove every in-edge into "to"
    run_select(monkeypatch, tmp_path, "- {t: 10.5, label: title, pre: 0.3, post: 0.3}\n")
    cap = capsys.readouterr()
    assert "1 inside speech" in cap.out and "WARNING" in cap.err

    with pytest.raises(SystemExit) as exc:
        run_select(monkeypatch, tmp_path, "- {t: 10.5, label: title, pre: 0.3, post: 0.3}\n", "--strict")
    assert exc.value.code == 2


def test_main_select_transcribes_in_short_chunks_unless_told_otherwise(stub_media, monkeypatch, tmp_path):
    # 120 s chunks dropped a third of the words on a recording with music under the voice; 30 s chunks heard them all.
    seen = []

    def asr(src, work, model, chunk=120):
        seen.append(chunk)
        return WORDS, 1.0

    monkeypatch.setattr(v, "transcribe", asr)
    (tmp_path / "moments.yaml").write_text("- {t: 20.0, label: quiet bit}\n")
    base = ["vidtight", str(tmp_path / "clip.mp4"), "--out", str(tmp_path / "out")]
    for extra in (["--select", str(tmp_path / "moments.yaml")], ["--select", str(tmp_path / "moments.yaml"), "--chunk", "20"], []):
        monkeypatch.setattr(sys, "argv", base + extra)
        v.main()
    assert seen == [30, 20.0, 120]  # select default, explicit, and the tighten mode's unchanged default


def test_transcribe_passes_the_chunk_length_to_parakeet(monkeypatch, tmp_path):
    from types import SimpleNamespace

    seen = {}

    class Model:
        def transcribe(self, wav, **kw):
            seen.update(kw)
            return SimpleNamespace(sentences=[])

    monkeypatch.setitem(sys.modules, "parakeet_mlx", SimpleNamespace(from_pretrained=lambda model: Model()))
    monkeypatch.setattr(v, "sh", lambda cmd: "")
    v.transcribe("clip.mov", tmp_path, "some/model", 30)
    assert seen == {"chunk_duration": 30, "overlap_duration": 5}
    v.transcribe("clip.mov", tmp_path, "some/model")
    assert seen == {"chunk_duration": 120, "overlap_duration": 15}


def test_main_select_with_render_reports_an_empty_selection_instead_of_crashing(stub_media, monkeypatch, tmp_path, capsys):
    def no_render(*a, **k):
        raise AssertionError("nothing was selected: the renderer must not run")

    monkeypatch.setattr(v, "render_audio", no_render)
    monkeypatch.setattr(v, "render_video", no_render)
    (tmp_path / "speech.yaml").write_text(f'- [0, {DUR}, "one unbroken line"]\n')  # no gap anywhere: the only moment has to be dropped
    with pytest.raises(SystemExit) as exc:
        run_select(monkeypatch, tmp_path, "- {t: 30.0, label: mid-line}\n", "--speech", str(tmp_path / "speech.yaml"), "--render")
    assert "nothing to render" in str(exc.value.code) and "select.csv" in str(exc.value.code)
    out = tmp_path / "out"
    assert json.loads((out / "edl.json").read_text())["keep"] == []  # the analysis outputs are still written
    (row,) = csv.DictReader((out / "select.csv").open())
    assert row["status"] == "dropped" and "--max-clip" in row["note"]
    assert "select: 1 moments -> 0 clips" in capsys.readouterr().out
    run_select(monkeypatch, tmp_path, "- {t: 30.0, label: mid-line}\n", "--speech", str(tmp_path / "speech.yaml"))  # analysis only: a normal exit


def test_speech_without_select_is_an_error(stub_media, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["vidtight", "clip.mp4", "--out", str(tmp_path), "--speech", "speech.yaml"])
    with pytest.raises(SystemExit):
        v.main()


# --- b-roll and storyboard ---------------------------------------------------------------------


def test_broll_windows_of_a_silent_file_are_the_whole_file_even_when_it_is_short():
    # Pause cutting would keep only the pads at each end. A silent hero shot is one window.
    (hero,) = v.broll_windows([], 24.0)
    assert (hero["start"], hero["end"], hero["duration"]) == (0.0, 24.0, 24.0)
    (short,) = v.broll_windows([], 0.4, minlen=0.8)
    assert short["duration"] == pytest.approx(0.4)


def test_broll_windows_drop_setup_chatter_and_the_breaths_inside_it():
    (shot,) = v.broll_windows(v.speech_blocks(say("hang on a second.", 0.0)), 24.0)
    assert shot["start"] == pytest.approx(1.2) and shot["end"] == pytest.approx(24.0)
    words = say("Auto", 1.0) + say("Jack.", 1.46) + say("Hi there.", 2.0) + say("Bye.", 4.0)  # 0.16 s and 0.24 s gaps inside the chatter
    lengths = [round(w["duration"], 2) for w in v.broll_windows(v.speech_blocks(words), DUR)]
    assert 0.16 not in lengths and 0.24 not in lengths
    assert any(d > 1 for d in lengths)


def test_a_broll_moment_keeps_the_whole_silent_window_not_a_pre_post_slice():
    blocks, gaps = gaps_for(WORDS)
    clips, (row,) = v.select_broll([{"t": 20.0, "label": "hero", "section": "chorus", "note": "hold the wide shot"}], v.windows_of(gaps, 0.8), blocks, DUR)
    assert len(clips) == 1 and clips[0][1] - clips[0][0] > 12  # the gap around t=20, not the 5 s default window
    assert clips[0][0] <= 20 <= clips[0][1]
    assert row["status"] == "kept" and row["section"] == "chorus" and row["shot_note"] == "hold the wide shot"


def test_broll_without_a_selection_keeps_every_window_and_an_empty_list_keeps_none():
    windows = v.broll_windows([], DUR)
    clips, rows = v.select_broll(None, windows, [], DUR)
    assert clips == [[0.0, DUR]] and rows == []
    assert v.select_broll([], windows, [], DUR) == ([], [])


def test_an_empty_moments_file_with_broll_keeps_nothing(stub_media, monkeypatch, tmp_path):
    for text in ("[]\n", "moments: []\n"):
        out = run_select(monkeypatch, tmp_path, text, "--broll", "--snap", "0")
        edl = json.loads((out / "edl.json").read_text())
        assert edl["keep"] == [] and edl["select"] == []
        assert json.loads((out / "shots.json").read_text()) == []
        assert json.loads((out / "windows.json").read_text())  # the speech-free inventory is still written


def test_broll_drops_a_moment_that_lands_in_speech():
    blocks, gaps = gaps_for(WORDS)
    clips, (row,) = v.select_broll([{"t": 10.5, "label": "inside the line"}], v.windows_of(gaps, 0.8), blocks, DUR)
    assert clips == [] and row["status"] == "dropped" and "speech" in row["note"]


def test_moments_keep_an_optional_section_and_note_without_changing_the_cut():
    _, (row,) = pick([{"t": 20.0, "label": "quiet bit", "section": "verse", "note": "let it breathe"}])
    assert row["status"] == "kept" and row["section"] == "verse" and row["shot_note"] == "let it breathe"
    assert row["note"] == ""  # the caller's note does not replace the edge-move note


def test_write_srt_is_empty_when_the_keep_holds_no_words(tmp_path):
    out = tmp_path / "o.srt"
    assert v.write_srt([{"w": "Hello.", "s": 0.0, "e": 0.4, "eos": True}], [[10.0, 12.0, 1]], out) == 0
    assert out.read_text() == ""


def _render_cmds(monkeypatch):
    cmds = []
    monkeypatch.setattr(v, "sh", lambda cmd: cmds.append(cmd) or "30/1,30/1\n")
    monkeypatch.setattr(v, "render_audio", lambda *a, **k: None)
    monkeypatch.setattr(v, "render_video", lambda *a, **k: None)
    return cmds


def test_broll_render_omits_an_empty_subtitle_track(stub_media, monkeypatch, tmp_path):
    cmds = _render_cmds(monkeypatch)
    out = tmp_path / "out"
    (tmp_path / "words.json").write_text("[]")
    monkeypatch.setattr(sys, "argv", ["vidtight", str(tmp_path / "clip.mp4"), "--out", str(out), "--words", str(tmp_path / "words.json"), "--broll", "--render"])
    v.main()
    mux = next(c for c in cmds if "+faststart" in c)
    assert str(out / "clip.srt") not in mux and "2:s" not in mux
    assert (out / "clip.srt").read_text() == ""


def test_render_still_muxes_subtitles_when_the_cut_has_speech(stub_media, monkeypatch, tmp_path):
    cmds = _render_cmds(monkeypatch)
    out = run_select(monkeypatch, tmp_path, "- {t: 10.5, label: title, pre: 0.3, post: 0.3}\n", "--render", "--snap", "0")
    mux = next(c for c in cmds if "+faststart" in c)
    assert str(out / "clip.srt") in mux and "2:s" in mux


def test_main_broll_keeps_a_silent_hero_shot_whole(stub_media, monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"
    (tmp_path / "words.json").write_text("[]")
    monkeypatch.setattr(sys, "argv", ["vidtight", str(tmp_path / "clip.mp4"), "--out", str(out), "--words", str(tmp_path / "words.json"), "--broll"])
    v.main()
    windows = json.loads((out / "windows.json").read_text())
    assert windows == [{"start": 0.0, "end": 60.0, "duration": 60.0}]
    edl = json.loads((out / "edl.json").read_text())
    assert edl["keep"] == [[0.0, 60.0, 1]] and edl["windows"] == windows
    assert not any(c[2] == "pause" for c in edl["cuts"])
    assert "b-roll: 1 speech-free window" in capsys.readouterr().out
    assert not (out / "select.csv").exists()


def test_main_broll_leaves_setup_chatter_out_of_the_window(stub_media, monkeypatch, tmp_path):
    out = tmp_path / "out"
    (tmp_path / "words.json").write_text(json.dumps(say("hang on a second.", 0.0)))
    monkeypatch.setattr(sys, "argv", ["vidtight", str(tmp_path / "clip.mp4"), "--out", str(out), "--words", str(tmp_path / "words.json"), "--broll", "--snap", "0"])
    v.main()
    (window,) = json.loads((out / "windows.json").read_text())
    assert window["start"] == pytest.approx(1.2) and window["duration"] > 50
    (s, e, _), = json.loads((out / "edl.json").read_text())["keep"]
    assert s > 1 and e - s > 50  # the hero shot, not the chatter plus a pad


def test_main_select_broll_keeps_the_silent_window_around_the_moment(stub_media, monkeypatch, tmp_path):
    moments = '- {t: 20.0, label: hero, section: chorus, note: "hold the wide shot"}\n'
    out = run_select(monkeypatch, tmp_path, moments, "--broll", "--snap", "0")
    (s, e, _), = json.loads((out / "edl.json").read_text())["keep"]
    assert e - s > 12 and s <= 20 <= e  # not the default 19-24 slice
    assert (out / "windows.json").exists() and (out / "select.csv").exists()
    (shot,) = json.loads((out / "shots.json").read_text())
    assert shot["section"] == "chorus" and shot["note"] == "hold the wide shot" and shot["status"] == "kept"
    assert "## chorus" in (out / "storyboard.md").read_text() and "hold the wide shot" in (out / "storyboard.md").read_text()


def test_main_select_writes_a_storyboard_grouped_by_section(stub_media, monkeypatch, tmp_path):
    moments = '- {t: 20.0, label: "quiet bit", section: verse, note: "let it breathe"}\n- {t: 99.0, label: nope, section: end}\n'
    out = run_select(monkeypatch, tmp_path, moments, "--snap", "0")
    shots = json.loads((out / "shots.json").read_text())
    assert [(s["label"], s["section"], s["note"], s["status"]) for s in shots] == [
        ("quiet bit", "verse", "let it breathe", "kept"),
        ("nope", "end", "", "dropped"),
    ]
    assert shots[0]["duration"] == pytest.approx(5.0)
    md = (out / "storyboard.md").read_text()
    assert md.index("## verse") < md.index("## end") and "let it breathe" in md
    rows = list(csv.DictReader((out / "select.csv").open()))
    assert rows[0]["section"] == "verse" and rows[0]["shot_note"] == "let it breathe" and rows[0]["note"] == ""
