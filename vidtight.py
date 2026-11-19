#!/usr/bin/env python3
"""vidtight: local screen-recording tightener. ASR (parakeet-mlx) -> fillers/pauses/retakes -> vocab fix -> EDL -> SRT + ffmpeg render.
--select turns it round: keep only the windows around wanted moments, with every edge in a speech gap."""
import argparse, csv, difflib, json, re, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
import yaml

FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm", "umm"}
SOFT_START = {"so", "like", "basically", "okay", "ok", "right"}  # filler only at sentence start
PHRASES = [("you", "know"), ("kind", "of"), ("sort", "of")]  # ("i","mean") opt-in via --cut-i-mean

LEAD = 0.0  # seconds of title card prepended (set by --leadin)

# --select edge rules
MINPAD, MAXPAD = 0.03, 0.2  # an edge sits 30-200 ms from the nearest word; under 30 ms counts as unsafe
TIGHT = 1.0  # a tight gap (narrower than --gap, between two exact --speech intervals) is used only when it saves this many seconds of edge movement over a clean one
KEEPMIN = 0.5  # drop a moment rather than keep less than this share of the window it asked for
JOIN = 0.5  # clips closer together than this are merged into one

def sh(cmd): return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

def duration(p): return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))

def norm(w): return re.sub(r"[^\w']", "", w.lower())

def wav16k(src, wav): sh(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)])

def probe_fps(src):
    """Frame rate of the edit grid. VFR captures (Quest: 100000/1) report a tick rate as r_frame_rate; use the average rate there."""
    r, _, avg = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-of", "csv=p=0", str(src)]).strip().splitlines()[0].partition(",")
    return round(Fraction(avg if Fraction(r) > 240 and avg and not avg.endswith("/0") else r))

# (a)+(b) audio + local ASR with word timestamps
def transcribe(src, work, model, chunk=120):
    wav = work / "audio.wav"; wav16k(src, wav)
    from parakeet_mlx import from_pretrained
    t = time.time(); m = from_pretrained(model)
    res = m.transcribe(str(wav), chunk_duration=chunk, overlap_duration=min(15, chunk / 6))
    words = []
    for s in res.sentences:
        for tok in s.tokens:
            if tok.text.startswith(" ") or not words or words[-1].get("eos"):
                words.append({"w": tok.text.strip(), "s": tok.start, "e": tok.end})
            else:
                words[-1]["w"] += tok.text; words[-1]["e"] = tok.end
        if words: words[-1]["eos"] = True
    return [w for w in words if w["w"]], time.time() - t

# (c) dead space: ffmpeg silencedetect is ground truth (ASR word ends smear across silences)
def silences(src, noise, mind):
    err = subprocess.run(["ffmpeg", "-v", "info", "-i", str(src), "-vn", "-af", f"silencedetect=n={noise}dB:d={mind}", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    return list(zip(st, en + [None] * (len(st) - len(en))))

# (f) vocab correction: explicit variants + fuzzy n-gram match against canonical terms
def apply_vocab(words, vocab):
    table = []
    for canon, variants in vocab.items():
        for v in [canon] + list(variants or []):
            table.append((tuple(norm(x) for x in v.split()), canon))
    if not table: return []  # empty vocab: nothing to fix (the fuzzy pass below needs at least one term)
    fixes = []
    i = 0
    while i < len(words):
        for n in (3, 2, 1):
            gram = tuple(norm(w["w"]) for w in words[i:i + n])
            if len(gram) < n: continue
            hit = next((c for v, c in table if v == gram), None)
            if not hit:
                g = " ".join(gram)
                best = max(table, key=lambda t: difflib.SequenceMatcher(None, " ".join(t[0]), g).ratio())
                if len(best[0]) == n and len(g) >= 6 and difflib.SequenceMatcher(None, " ".join(best[0]), g).ratio() >= 0.85:
                    hit = best[1]
            if hit:
                old = " ".join(w["w"] for w in words[i:i + n])
                suf = re.search(r"(['’]s)?[.,!?;:]*$", words[i + n - 1]["w"]).group(0)  # keep possessive + punctuation
                new = hit + suf
                if old != new: fixes.append({"t": words[i]["s"], "from": old, "to": new})
                words[i] = {**words[i], "w": new, "e": words[i + n - 1]["e"], "eos": words[i + n - 1].get("eos")}
                del words[i + 1:i + n]
                break
        i += 1
    return fixes

def sentences(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if w.get("eos") or re.search(r"[.!?]$", w["w"]): out.append(cur); cur = []
    if cur: out.append(cur)
    return out

# (d) fillers (word level) + (e) retakes
def find_cuts(words, a):
    cuts = []
    sents = sentences(words)
    for si, s in enumerate(sents):
        for j, w in enumerate(s):
            n = norm(w["w"])
            if n in FILLERS or (j == 0 and n in SOFT_START and len(s) > 3):
                cuts.append((w["s"], w["e"], "filler", w["w"])); w["cut"] = True
            if j and n and n == norm(s[j - 1]["w"]) and n not in {"that", "had"}:
                cuts.append((s[j - 1]["s"], s[j - 1]["e"], "stutter", w["w"])); s[j - 1]["cut"] = True
            if j + 1 < len(s) and (n, norm(s[j + 1]["w"])) in PHRASES:
                cuts.append((w["s"], s[j + 1]["e"], "filler", w["w"] + " " + s[j + 1]["w"])); w["cut"] = s[j + 1]["cut"] = True
    # retake: a run of >=N words repeated shortly after -> cut from first copy to start of second (keep last take)
    sil = [(s0, a.dur if e0 is None else e0) for s0, e0 in silences(a.wav, a.noise, 0.3)]
    seq = [w for w in words if not w.get("cut")]
    toks = [norm(w["w"]) for w in seq]; N = a.retake_words; i = 0
    while i + N <= len(toks):
        for j in range(i + N, min(i + N + 12, len(toks) - N + 1)):
            if toks[i:i + N] == toks[j:j + N]:
                end = max(seq[j - 1]["e"], seq[j]["s"] - 0.05)
                nxt = seq[j + 1]["s"] if j + 1 < len(seq) else a.dur
                # ASR often glues the aborted take's last word onto the new take; trust the silence that splits them
                end = next((e0 - 0.03 for s0, e0 in sil if seq[j]["s"] - 0.1 <= e0 <= nxt + 0.3), end)
                cuts.append((seq[i]["s"], end, "retake", " ".join(toks[i:j])[:70]))
                for w in seq[i:j]: w["cut"] = True
                i = j; break
        else: i += 1
    for s0, e0 in silences(a.wav, a.noise, a.pause):
        e0 = a.dur if e0 is None else e0
        if e0 - s0 - 2 * a.pad >= a.mincut: cuts.append((s0 + a.pad, e0 - a.pad, "pause", f"{e0 - s0:.1f}s"))
    return sorted(cuts)

# (g) EDL: invert cuts into keep segments [s, e, speed]
def build_edl(cuts, dur):
    merged = []
    for s, e, *_ in sorted(cuts):
        s, e = max(0, s), min(dur, e)
        if e <= s: continue
        if merged and s <= merged[-1][1] + 0.05: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    keep, t = [], 0.0
    for s, e in merged:
        if s - t > 0.05: keep.append([t, s, 1])
        t = e
    if dur - t > 0.05: keep.append([t, dur, 1])
    return keep

def load_rms(wav):
    import numpy as np, wave
    with wave.open(str(wav)) as f: x = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32)
    hop = 160
    return np.sqrt(np.convolve((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1), np.ones(3) / 3, "same"))

def quiet(rms, a, b):  # time of quietest 10ms frame in [a, b]
    import numpy as np
    i0, i1 = max(0, int(a * 100)), min(len(rms), int(b * 100) + 1)
    return (i0 + int(np.argmin(rms[i0:i1]))) / 100 if i1 > i0 else (a + b) / 2

def fix_fillers(cuts, rms):
    """Parakeet stamps fillers late: move a filler cut's start back to the dip *before* the filler."""
    out = []
    for c in cuts:
        if c[2] in ("filler", "stutter"): c = (quiet(rms, c[0] - 0.3, c[0] - 0.05), c[1], c[2], c[3])
        out.append(c)
    return out

def snap(keep, rms, win, lim=None):
    """Move each cut edge to the quietest frame within win (keep-end looks back, keep-start looks forward). lim(t) -> (lo, hi) fences the edge at t."""
    if win <= 0: return keep
    end = len(rms) / 100
    def q(t, a, b):
        if not lim: return quiet(rms, a, b)
        lo, hi = lim(t); a, b = max(a, lo), min(b, hi)
        return min(max(quiet(rms, a, b), a), b) if b > a else min(max(t, lo), hi)
    out = [[0.0 if s < 0.01 else q(s, s - 0.03, s + win), e if e >= end - 0.01 else q(e, e - win, e + 0.03), sp] for s, e, sp in keep]
    return [k for k in out if k[1] - k[0] > 0.05]

def apply_speed(keep, ranges):
    """Split keep segments so [a, b] plays at speed f."""
    for a0, b0, f in ranges:
        nk = []
        for s, e, sp in keep:
            if sp == 1 and s < a0 and b0 < e: nk += [[s, a0, 1], [a0, b0, f], [b0, e, 1]]
            else: nk.append([s, e, sp])
        keep = nk
    return keep

def quantize(keep, fps):
    q = lambda t: round(t * fps) / fps
    return [[q(s), q(e), sp] for s, e, sp in keep if q(e) > q(s)]

def seglen(k): return (k[1] - k[0]) / k[2]

def remap(t, keep):
    acc = 0.0
    for s, e, sp in keep:
        if t < s: return acc
        if t <= e: return acc + (t - s) / sp
        acc += (e - s) / sp
    return acc

def ts(t, sep=","): t = round(t, 3); h, r = divmod(t, 3600); m, s = divmod(r, 60); return f"{int(h):02}:{int(m):02}:{int(s):02}{sep}{int(round(s * 1000)) % 1000:03}"

# (h) retimed SRT from kept words
def write_srt(words, keep, path, replace=None, extra=(), maxw=8):
    kept = [w for w in words if not w.get("cut") and any(s <= (w["s"] + w["e"]) / 2 < e or s <= w["s"] < e for s, e, _ in keep)]
    lines, cur = [], []
    for w in kept:
        cur.append(w)
        if len(cur) >= maxw or w.get("eos") or re.search(r"[.!?]$", w["w"]): lines.append(cur); cur = []
    if cur: lines.append(cur)
    items = []
    for l in lines:
        s, e = remap(l[0]["s"], keep), remap(min(l[-1]["e"], l[-1]["s"] + 1.0), keep)
        txt = " ".join(w["w"] for w in l)
        for k, v in (replace or {}).items(): txt = txt.replace(k, v)
        items.append((s, max(e, s + 0.3), txt))
    items += list(extra)
    items = [(s + LEAD, e + LEAD, t) for s, e, t in items]
    out = [f"{i}\n{ts(s)} --> {ts(e)}\n{t}\n" for i, (s, e, t) in enumerate(sorted(items), 1)]
    Path(path).write_text("\n".join(out))

# (i) --select: keep the windows around wanted moments instead of cutting the unwanted bits. Every edge goes in a speech gap.
def wordcap(w, maxword=1.2): return min(maxword, 0.3 + 0.1 * len(norm(w)))  # longest a spoken word plausibly lasts: 0.3 s + 0.1 s a letter

def shorten(s, n=60): return s if len(s) <= n else s[:n // 2 - 1].rstrip() + " … " + s[2 - n // 2:].lstrip()

def speech_blocks(words, speech=(), sil=(), maxword=1.2):
    """Speech as uncuttable blocks {s, e, w, src, raw, eos}, sorted by start. Caller intervals [start, end, label] (src "app") are exact and replace
    the ASR words they overlap. ASR word ends smear into whatever follows the word, so each one is cut back to the first detected silence inside it
    (src "silence"), else to a plausible word length (src "cap": an estimate, not a measurement). raw keeps the end as stamped."""
    app = [{"s": float(x[0]), "e": float(x[1]), "w": str(x[2]) if len(x) > 2 else "speech", "src": "app", "raw": float(x[1]), "eos": True} for x in speech]
    out = list(app)
    for w in words:
        s, e, src = w["s"], w["e"], "asr"
        cut = next((a for a, _ in sil if s + 0.05 < a < e), None)
        if cut is not None: e, src = cut, "silence"
        if e - s > wordcap(w["w"], maxword): e, src = s + wordcap(w["w"], maxword), "cap"
        if any(b["s"] < e and s < b["e"] for b in app): continue
        out.append({"s": s, "e": max(e, s + 0.01), "w": w["w"], "src": src, "raw": w["e"], "eos": bool(w.get("eos") or re.search(r"[.!?]$", w["w"]))})
    return sorted(out, key=lambda b: (b["s"], b["e"]))

def utterances(blocks, gap=0.4):
    """Blocks joined into utterances {s, e, text, est} wherever the pause between them is under gap. est counts the estimated word ends inside."""
    out = []
    for b in blocks:
        if out and b["s"] - out[-1]["e"] < gap - 1e-9: u = out[-1]; u["e"] = max(u["e"], b["e"]); u["text"] += " " + b["w"]; u["est"] += b["src"] == "cap"
        else: out.append({"s": b["s"], "e": b["e"], "text": b["w"], "est": int(b["src"] == "cap")})
    return out

def speech_gaps(blocks, dur, gap=0.4, mingap=0.15, tail=0.15, lead=MAXPAD, floor=MINPAD):
    """The gap before each block and after the last, each {s, e, prev, next, grade, lo, hi}. grade is "clean" (gap s or more, or open to the start or
    end of the file), "tight" (mingap s or more, and only between two caller intervals: ASR word times drift by more than a tight gap is wide, so a
    short gap seen only by the ASR cannot be trusted) or None (not cuttable). An edge may sit anywhere in [lo, hi]: tail s after the word before the
    gap and lead s before the word after it (ASR can stamp a start late). A narrower gap gets one point, the pads scaled to fit, neither under floor."""
    out, end, prev = [], 0.0, None
    for b in [*blocks, None]:
        s = max(end, dur if b is None else b["s"]); w = s - end
        if prev is None or b is None: grade, lo, hi = "clean", (0.0 if prev is None else min(s, end + min(tail, w))), (s if b is None else max(end, s - min(lead, w)))
        else:
            grade = "clean" if w >= gap - 1e-9 else "tight" if w >= mingap - 1e-9 and prev["src"] == b["src"] == "app" else None
            lo, hi = (end + tail, s - lead) if w >= tail + lead else (end + min(max(w * tail / (tail + lead), floor), w - floor),) * 2
        out.append({"s": end, "e": s, "prev": prev, "next": b, "grade": grade, "lo": lo, "hi": hi})
        if b and b["e"] >= end: end, prev = b["e"], b
    return out

def select_moments(moments, gaps, blocks, dur, pre=1.0, post=4.0, maxclip=12.0):
    """A keep-window around each moment {t, label, pre?, post?, caption?} with both edges in a usable speech gap. A requested edge that lands in speech
    moves to a gap: outward takes the utterance whole (the clip may grow to maxclip), inward trims the window (never below KEEPMIN of it). The cheapest
    pair of moves wins, a tight gap costing TIGHT s extra; when no pair fits the moment is dropped. Returns (clips, rows): clips are [in, out] in source
    order, overlapping and near-touching ones merged; rows, one per moment in the caller's order, say what was done and why."""
    use = [g for g in gaps if g["grade"]]
    said = lambda x0, x1: " ".join(b["w"] for b in blocks if x0 <= (b["s"] + b["e"]) / 2 <= x1)  # speech a clip contains
    hit = lambda x0, x1: shorten(" ".join(b["w"] for b in blocks if b["s"] < x1 and x0 < b["e"]), 44)  # speech an edge moved across
    rows = []
    for m in moments:
        t = float(m["t"]); a0, b0 = max(0.0, t - float(m.get("pre", pre))), min(dur, t + float(m.get("post", post)))
        r = {"label": str(m.get("label", "")), "t": t, "want": [a0, b0], "status": "dropped", "clip": None, "in": None, "out": None, "note": "", "speech": "", "caption": m.get("caption")}
        rows.append(r)
        if not 0 <= t <= dur or b0 - a0 < 0.3: r["note"] = "t is outside the recording" if not 0 <= t <= dur else "the window is under 0.3 s"; continue
        cap = max(maxclip, b0 - a0) + 1e-6
        ins = [(x, g) for g in use for x in [min(max(a0, g["lo"]), g["hi"])] if x <= t]; outs = [(x, g) for g in use for x in [min(max(b0, g["lo"]), g["hi"])] if x >= t]
        fit = [(abs(i - a0) + abs(o - b0) + TIGHT * ((gi["grade"] == "tight") + (go["grade"] == "tight")), o - i, i, o)
               for i, gi in ins if i >= t - cap for o, go in outs if 0.3 <= o - i <= cap and min(o, b0) - max(i, a0) >= KEEPMIN * (b0 - a0) - 1e-6]
        if not fit:
            wi, wo = max(x for x, _ in ins if x <= a0 + 0.25), min(x for x, _ in outs if x >= b0 - 0.25)  # the edges that would take all the speech whole
            r["note"] = f'the window cuts "{hit(wi, wo)}": taking it whole needs {wo - wi:.1f}s (--max-clip {maxclip:g}) and trimming to a gap would keep under half the window'
            continue
        _, _, i, o = min(fit)
        notes = []
        for side, d, x0, x1 in (("in", i - a0, min(i, a0), max(i, a0)), ("out", o - b0, min(o, b0), max(o, b0))):
            if abs(d) > 0.25: notes.append(f"{side} {d:+.2f}s to " + ("take in" if (d < 0) == (side == "in") else "leave out") + f' "{hit(x0, x1)}"')
            elif abs(d) > 0.005: notes.append(f'{side} {d:+.2f}s to clear "{hit(x0 - 0.25, x1 + 0.25)}"')
        wid, trim = i - a0 < -0.25 or o - b0 > 0.25, i - a0 > 0.25 or o - b0 < -0.25
        r.update({"in": i, "out": o, "note": "; ".join(notes), "status": "+".join(k for k, c in (("widened", wid), ("trimmed", trim)) if c) or "kept"})
    clips = []
    for r in sorted((r for r in rows if r["in"] is not None), key=lambda r: r["in"]):
        if clips and r["in"] <= clips[-1][1] + JOIN: clips[-1][1] = max(clips[-1][1], r["out"])
        else: clips.append([r["in"], r["out"]])
        r["clip"] = len(clips)
    for r in rows:
        if r["clip"]:
            r["in"], r["out"] = clips[r["clip"] - 1]; r["speech"] = said(r["in"], r["out"])
            if sum(x["clip"] == r["clip"] for x in rows) > 1: r["status"] += "+merged"
    return clips, rows

def check_edl(keep, blocks, dur, gap=0.4):
    """Re-read every edge of an EDL against the speech map. One dict per edge in keep order: {t, side, safe, inside, tight, est, why}. inside: within a
    word. safe: not inside and at least MINPAD from every word. tight: in a gap narrower than gap. est: safe only if an estimated word end is right,
    because the ASR stamped that word as still running at t."""
    out = []
    for s, e, *_ in keep:
        for side, x in (("in", s), ("out", e)):
            d = {"t": x, "side": side, "safe": True, "inside": False, "tight": False, "est": False, "why": "file start" if x <= 0.001 else "file end"}
            out.append(d)
            if x <= 0.001 or x >= dur - 0.05: continue
            word = next((b for b in blocks if b["s"] < x < b["e"]), None)
            if word: d.update(safe=False, inside=True, why=f'inside "{word["w"]}" ({word["s"]:.2f}-{word["e"]:.2f})'); continue
            prev, nxt = max((b for b in blocks if b["e"] <= x), key=lambda b: b["e"], default=None), min((b for b in blocks if b["s"] >= x), key=lambda b: b["s"], default=None)
            gs, ge = prev["e"] if prev else 0.0, nxt["s"] if nxt else dur
            near = min([x - gs] * bool(prev) + [ge - x] * bool(nxt), default=dur)
            d.update(safe=near >= MINPAD - 1e-6, tight=bool(prev and nxt and ge - gs < gap - 1e-9), est=bool(prev and prev["src"] == "cap" and x < prev["raw"]))
            why = ", ".join(f'{abs(x - at):.2f}s {rel} "{b["w"]}"' for b, at, rel in ((prev, gs, "after"), (nxt, ge, "before")) if b) or "no speech in the recording"
            if d["est"]: why += f' (the end of "{prev["w"]}" is an estimate; ASR stamped it to {prev["raw"]:.2f})'
            d["why"] = ("" if d["safe"] else f"under {MINPAD * 1000:.0f} ms from a word: ") + (f"{ge - gs:.2f}s gap: " if prev and nxt else "") + why
    return out

def select_edl(a, words, rms, fps):
    """--select: keep-segments around the wanted moments. Returns (keep, cuts, sel); sel carries the per-moment rows, the speech map and caption words."""
    y = yaml.safe_load(Path(a.select).expanduser().read_text()) or []
    speech = (yaml.safe_load(Path(a.speech).expanduser().read_text()) or []) if a.speech else []
    sil = [(s0, a.dur if e0 is None else e0) for s0, e0 in silences(a.wav, a.noise, 0.3)]
    blocks = speech_blocks(words, speech, sil, a.max_word); hf = 0.5 / fps  # half a frame of slack so frame quantising cannot eat a pad
    gaps = speech_gaps(blocks, a.dur, a.gap, max(a.min_gap, 0.12), min(max(a.pad, MINPAD), MAXPAD) + hf, MAXPAD, MINPAD + hf)
    clips, rows = select_moments(y.get("moments", []) if isinstance(y, dict) else y, gaps, blocks, a.dur, a.pre, a.post, a.max_clip)
    lim = lambda t: next(((g["lo"], g["hi"]) for g in gaps if g["s"] <= t <= g["e"]), (t, t))  # snapping stays inside the gap, pads intact
    keep = quantize(snap([[i, o, 1] for i, o in clips], rms, a.snap, lim), fps)
    if len(keep) != len(clips): sys.exit("vidtight: a selected clip collapsed while snapping; rerun with --snap 0")
    cuts, t0 = [], 0.0
    for s, e, _ in keep:
        if s - t0 > 1e-6: cuts.append((t0, s, "unselected", ""))
        t0 = e
    if a.dur - t0 > 1e-6: cuts.append((t0, a.dur, "unselected", ""))
    edges = check_edl(keep, blocks, a.dur, a.gap)
    for r in rows:
        if not r["clip"]: continue
        k = r["clip"] - 1; r["in"], r["out"] = keep[k][:2]
        for side, e in (("in", edges[2 * k]), ("out", edges[2 * k + 1])): r[side + "_safe"], r[side + "_why"] = e["safe"], e["why"]
    asked = check_edl([[*r["want"], 1] for r in rows if 0 <= r["t"] <= a.dur], blocks, a.dur, a.gap)
    capw = [{**w, "e": min(w["e"], w["s"] + wordcap(w["w"], a.max_word))} for w in words]  # captions must not ride a smeared word into another clip
    return keep, cuts, {"rows": rows, "blocks": blocks, "utts": utterances(blocks, a.gap), "words": capw, "asked": asked, "sil": len(sil)}

def report_select(a, work, sel):
    """Write select.csv, say what happened to each moment, then re-read the EDL on disk against the speech map. Returns the unsafe edges."""
    rows, blocks = sel["rows"], sel["blocks"]
    with open(work / "select.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["moment", "t", "status", "clip", "in", "out", "dur", "in_safe", "in_why", "out_safe", "out_why", "note", "speech", "caption"])
        for r in rows:
            c = r["clip"]; yn = lambda k: ("yes" if r[k] else "no") if c else ""
            w.writerow([r["label"], f"{r['t']:.3f}", r["status"], c or "", *([f"{r['in']:.3f}", f"{r['out']:.3f}", f"{r['out'] - r['in']:.3f}"] if c else [""] * 3),
                        yn("in_safe"), r.get("in_why", ""), yn("out_safe"), r.get("out_why", ""), r["note"], r["speech"], r["caption"] or ""])
    est, app, bysil = [b for b in blocks if b["src"] == "cap"], sum(b["src"] == "app" for b in blocks), sum(b["src"] == "silence" for b in blocks)
    print(f"speech map: {len(blocks) - app} ASR words + {app} caller intervals -> {len(sel['utts'])} utterances; silencedetect found {sel['sil']} pause(s), which set {bysil} word end(s)")
    if est:
        worst, lost = max(est, key=lambda b: b["raw"] - b["s"]), sum(b["raw"] - b["s"] >= 3 for b in est)
        print(f"speech map: LOW CONFIDENCE. {len(est)} of {len(blocks) - app} word ends are estimates: ASR stamped those words longer than a word can last "
              f"(worst: \"{worst['w']}\" at {worst['s']:.2f}s, stamped {worst['raw'] - worst['s']:.2f}s long) and no detected silence shows where they really end. "
              "That is what music or room tone under the voice does. Gaps after those words are inferred from a length cap, not measured: listen to those edges.")
        if lost: print(f"speech map: {lost} of those words are stamped 3 s or longer, the mark of the ASR losing its place. Lines spoken after them may be missing from the transcript altogether, "
                       "and no check here can see an edge inside speech that was never transcribed. Re-transcribe in short chunks (drop --words/--reuse, or lower --chunk) or pass the app's own --speech intervals.")
    n = {k: sum(k in r["status"] for r in rows) for k in ("kept", "widened", "trimmed", "merged", "dropped")}
    print(f"select: {len(rows)} moments -> {len({r['clip'] for r in rows if r['clip']})} clips ({n['kept']} as asked, {n['widened']} widened, {n['trimmed']} trimmed, {n['merged']} merged, {n['dropped']} dropped)")
    for i, r in enumerate(rows, 1):
        span = f"{r['in']:.2f}-{r['out']:.2f} ({r['out'] - r['in']:.2f}s, clip {r['clip']})" if r["clip"] else "-"
        print(f"  {i}. {r['status']:<15} {span}  {r['label']}" + (f"  [{r['note']}]" if r["note"] else ""))
    asked = sel["asked"]
    print(f"as asked, {sum(e['inside'] for e in asked)} of {len(asked)} window edges were inside speech")
    edges = check_edl(json.loads((work / "edl.json").read_text())["keep"], blocks, a.dur, a.gap)
    bad = [e for e in edges if not e["safe"]]
    print(f"self-check: {len(edges)} edges re-read from edl.json: {sum(e['inside'] for e in edges)} inside speech, {len(bad) - sum(e['inside'] for e in edges)} under {MINPAD * 1000:.0f} ms from a word, "
          f"{sum(e['tight'] for e in edges)} in tight gaps (under {a.gap:g}s), {sum(e['est'] for e in edges)} rest on an estimated word end")
    if bad:
        print(f"WARNING: {len(bad)} EDGE(S) NOT SPEECH-SAFE. Do not ship this cut.", file=sys.stderr)
        for e in bad: print(f"  {e['side']} edge at {e['t']:.3f}s: {e['why']}", file=sys.stderr)
    return bad

# render: audio in one pass (fades, music, loudnorm), video in chunks (hw encode), then mux
def render_audio(src, keep, out, music=None, xf=0.02):
    """Assemble the voice edit in numpy (fast, sample-exact), then one ffmpeg pass for music + loudnorm."""
    import numpy as np, wave
    raw = Path(out).with_suffix(".src.s16")
    if not raw.exists(): sh(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", "48000", "-f", "s16le", str(raw)])
    X = np.memmap(raw, np.int16, "r").reshape(-1, 2); R = 48000
    parts = [np.zeros((int(round(LEAD * R)), 2), np.float32)] if LEAD else []
    for s, e, sp in keep:
        n = int(round(seglen([s, e, sp]) * R))
        if sp != 1: parts.append(np.zeros((n, 2), np.float32)); continue
        y = X[int(round(s * R)):int(round(s * R)) + n].astype(np.float32)
        f = min(int(xf * R), len(y) // 3)
        if f: r = np.linspace(0, 1, f)[:, None]; y[:f] *= r; y[-f:] *= r[::-1]
        parts.append(y)
    voice = Path(out).with_suffix(".voice.wav")
    with wave.open(str(voice), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(R)
        w.writeframes(np.clip(np.concatenate(parts), -32768, 32767).astype(np.int16).tobytes())
    inputs, fg = ["-i", str(voice)], ""
    if music:
        inputs += ["-i", str(music["file"])]
        st, slam, duck = music["start_out"] + LEAD, music["slam_out"] + LEAD, 10 ** (music.get("duck_db", -19) / 20)
        fg = (f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,"
              f"volume='{duck}+(1-{duck})*clip((t-{slam - st:.3f})/0.5,0,1)':eval=frame,afade=t=out:st={music['dur'] - music.get('afade', 1.5):.3f}:d={music.get('afade', 1.5):.3f},"
              f"adelay={int(st * 1000)}:all=1[mus];[0:a]apad=whole_dur={st + music['dur']:.3f}[vp];"
              f"[vp][mus]amix=inputs=2:normalize=0:duration=longest,")
    else: fg = "[0:a]"
    fg += "loudnorm=I=-12.5:TP=-1.5:LRA=11,aresample=48000,alimiter=limit=0.83:level=false" + (f",afade=t=out:st={music['start_out'] + LEAD + music['dur'] - 2.5:.3f}:d=2.5" if music else "") + "[aout]"
    sh(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fg, "-map", "[aout]", "-c:a", "aac_at", "-b:a", "256k", str(out)])

def render_video(src, keep, out, fps, bitrate, tail=0.0, fade=0.0, chunk=300.0, leadin=None, fadein=0.0):
    work = Path(out).parent / "vchunks"; work.mkdir(exist_ok=True)
    groups, cur, acc = [], [], 0.0
    for k in keep:
        cur.append(k); acc += k[1] - k[0]
        if acc >= chunk: groups.append(cur); cur, acc = [], 0.0
    if cur: groups.append(cur)
    files = []
    for gi, g in enumerate(groups):
        base = max(0.0, g[0][0] - 1.0)
        parts, cat = [], ""
        for i, (s, e, sp) in enumerate(g):
            parts.append(f"[0:v]trim={s - base:.6f}:{e - base:.6f},setpts=(PTS-STARTPTS)/{sp},fps={fps}[v{i}];"); cat += f"[v{i}]"
        fg = "".join(parts) + f"{cat}concat=n={len(g)}:v=1:a=0"
        if gi == 0 and fadein > 0: fg += f",fade=t=in:st=0:d={fadein:.3f}"
        if gi == len(groups) - 1 and tail > 0:
            L = sum(seglen(k) for k in g)
            fg += f",tpad=stop_mode=clone:stop_duration={tail:.3f},fade=t=out:st={L + tail - fade:.3f}:d={fade:.3f}"
        fg += "[vout]"
        f = work / f"c{gi:03}.mp4"; fgf = work / f"c{gi:03}.txt"; fgf.write_text(fg)
        end = g[-1][1] - base + 1.0
        sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{base:.6f}", "-t", f"{end:.6f}", "-i", str(src), "-/filter_complex", str(fgf), "-map", "[vout]",
            "-an", "-c:v", "h264_videotoolbox", "-b:v", bitrate, "-maxrate", bitrate, "-g", str(fps * 2), "-pix_fmt", "yuv420p", str(f)])
        files.append(f); print(f"  video chunk {gi + 1}/{len(groups)} done", flush=True)
    if leadin: files.insert(0, Path(leadin))
    lst = work / "list.txt"; lst.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    sh(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])

def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src"); p.add_argument("--vocab", help="vocabulary YAML (default: vocab.yaml next to vidtight.py, if present)")
    p.add_argument("--pause", type=float, default=0.8, help="cut silences longer than this (s)")
    p.add_argument("--pad", type=float, default=0.15, help="keep this much air around speech (s)")
    p.add_argument("--mincut", type=float, default=0.25); p.add_argument("--retake-words", type=int, default=5, help="min repeated words to call a retake")
    p.add_argument("--noise", type=float, default=-35, help="silence threshold dB")
    p.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v3")
    p.add_argument("--render", action="store_true"); p.add_argument("--name", help="output basename")
    p.add_argument("--out", help="output dir (default: next to source)")
    p.add_argument("--edits", help="edits.yaml: manual cuts, kept retakes, speed-ups, caption replacements, music")
    p.add_argument("--cut-i-mean", action="store_true")
    p.add_argument("--snap", type=float, default=0.08, help="edge snap window s (0 = off)")
    p.add_argument("--bitrate", default="16M")
    p.add_argument("--leadin", help="pre-encoded title card clip (same size/fps/codec) to prepend")
    p.add_argument("--fadein", type=float, default=0.0, help="fade first footage in from black (s)")
    p.add_argument("--reuse", action="store_true", help="reuse cached transcript")
    p.add_argument("--words", help="use this words.json transcript instead of running ASR")
    p.add_argument("--chunk", type=float, help="ASR chunk length s (default 120; 30 with --select, because long chunks drop whole lines when music runs under the voice)")
    p.add_argument("--select", help="moments.yaml [{t, label, pre?, post?, caption?}]: keep only the windows around these moments, every edge in a speech gap")
    p.add_argument("--speech", help="--select: exact speech intervals [[start, end, label], ...] logged by the app; they replace the ASR words they overlap")
    p.add_argument("--pre", type=float, default=1.0, help="--select: seconds kept before each moment"); p.add_argument("--post", type=float, default=4.0, help="--select: seconds kept after each moment")
    p.add_argument("--max-clip", type=float, default=12.0, help="--select: longest a clip may grow to take an utterance whole (s)")
    p.add_argument("--gap", type=float, default=0.4, help="--select: smallest speech gap an edge may use (s)")
    p.add_argument("--min-gap", type=float, default=0.15, help="--select: smallest gap usable between two exact --speech intervals (s)")
    p.add_argument("--max-word", type=float, default=1.2, help="--select: longest a single word can last; longer ASR stamps are cut back (s)")
    p.add_argument("--strict", action="store_true", help="--select: exit 2 if any edge of the final EDL is not speech-safe")
    a = p.parse_args()
    if a.speech and not a.select: p.error("--speech needs --select")
    global LEAD
    if a.leadin: LEAD = duration(a.leadin)
    src = Path(a.src).expanduser().resolve(); work = Path(a.out).expanduser() if a.out else src.parent / (src.stem + ".vidtight"); work.mkdir(parents=True, exist_ok=True)
    if a.cut_i_mean: PHRASES.append(("i", "mean"))
    a.dur = duration(src); a.src = src; a.wav = work / "audio.wav"
    fps = probe_fps(src)
    ed = yaml.safe_load(open(a.edits)) if a.edits else {}
    tj = work / "words.json"
    if a.words: words = json.loads(Path(a.words).expanduser().read_text()); tj.write_text(json.dumps(words))
    elif a.reuse and tj.exists(): words = json.loads(tj.read_text())
    else:
        chunk = a.chunk or (30 if a.select else 120)
        words, asr_t = transcribe(src, work, a.model, *([chunk] if chunk != 120 else [])); tj.write_text(json.dumps(words))
        print(f"ASR: {a.dur:.1f}s audio in {asr_t:.1f}s ({a.dur / asr_t:.0f}x realtime)")
    if not a.wav.exists(): wav16k(src, a.wav)  # --words or --reuse into a fresh --out dir
    vf = Path(a.vocab).expanduser() if a.vocab else Path(__file__).with_name("vocab.yaml")  # the bundled default is absent in a non-editable install
    fixes = apply_vocab(words, (yaml.safe_load(vf.read_text()) or {}) if a.vocab or vf.exists() else {})
    rms = load_rms(a.wav)
    speed, sel = [], None
    if a.select: keep, cuts, sel = select_edl(a, words, rms, fps)
    else:
        cuts = fix_fillers(find_cuts(words, a), rms)
        # manual edits
        keepr = ed.get("keep_retakes", [])
        dropped = [c for c in cuts if c[2] == "retake" and any(abs(c[0] - t) < 2 for t in keepr)]
        for c in dropped:
            for w in words:
                if c[0] <= w["s"] < c[1]: w["cut"] = False
        cuts = [c for c in cuts if c not in dropped]
        for t, f in ed.get("speed", []):
            pc = next((c for c in cuts if c[2] == "pause" and c[0] - 1 <= t <= c[1]), None)
            if pc: cuts.remove(pc); speed.append((pc[0], pc[1], f))
        for s0, e0, why in ed.get("cut", []):
            cuts.append((quiet(rms, s0 - 0.05, s0 + 0.05), quiet(rms, e0 - 0.05, e0 + 0.05), "manual", why))
            for w in words:
                if s0 <= w["s"] < e0: w["cut"] = True
        keep = snap(build_edl(cuts, a.dur), rms, a.snap)
        keep = quantize(apply_speed(keep, speed), fps)
    outdur = sum(seglen(k) for k in keep)
    music, extra, tail = ed.get("music"), [], 0.0
    if music:
        music["dur"] = duration(music["file"])
        music["start_out"], music["slam_out"] = remap(music["start_src"], keep), remap(music["slam_src"], keep)
        tail = max(0.0, music["start_out"] + music["dur"] - outdur)
        extra = [(music["slam_out"] + 0.3, music["start_out"] + music["dur"], music.get("caption", "\u266a music \u266a"))]
    if sel: extra += [(c0, max(c1, c0 + 0.3), str(r["caption"])) for r in sel["rows"] if r["clip"] and r["caption"] for c0, c1 in [(remap(max(r["in"], r["want"][0]), keep), remap(min(r["out"], r["want"][1]), keep))]]
    name = a.name or src.stem
    srt = work / f"{name}.srt"
    write_srt(sel["words"] if sel else words, keep, srt, ed.get("replace"), extra)
    edl = {"src": str(src), "keep": keep, "cuts": cuts, "speed": speed, "vocab_fixes": fixes}
    if sel: edl["select"], edl["speech"] = sel["rows"], sel["utts"]
    (work / "edl.json").write_text(json.dumps(edl, indent=1))
    with open(work / "cuts.csv", "w") as f:
        f.write("start,end,type,detail\n")
        for s, e, k, d in sorted(cuts): f.write(f"{ts(s, '.')},{ts(e, '.')},{k},\"{d}\"\n")
        for s, e, sp in speed: f.write(f"{ts(s, '.')},{ts(e, '.')},speed x{sp},\"{e - s:.1f}s -> {(e - s) / sp:.1f}s\"\n")
    by = {}
    for c in cuts: by[c[2]] = by.get(c[2], 0) + 1
    unsafe = report_select(a, work, sel) if sel else []
    if not sel: print(f"cuts: {by} speed-ups: {len(speed)} vocab fixes: {len(fixes)}")
    print(f"duration: {a.dur:.1f}s -> {outdur:.1f}s edit (+{tail:.1f}s music tail) = {outdur + tail:.1f}s (+{LEAD:.1f}s lead-in), {len(keep)} segments")
    if unsafe and a.strict: print(f"outputs in {work}"); sys.exit(2)
    if a.render:
        t = time.time()
        aud, vid, out = work / f"{name}.audio.m4a", work / f"{name}.video.mp4", work / f"{name}.mp4"
        render_audio(src, keep, aud, music); print(f"audio done {time.time() - t:.0f}s", flush=True)
        render_video(src, keep, vid, fps, a.bitrate, tail, fade=music.get("fade", tail) if music else 0.0, leadin=a.leadin, fadein=a.fadein)
        # end with the video, not -shortest: the caption track stops at the last spoken word, which would cut off any footage after it
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-i", str(aud), "-i", str(srt), "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-t", f"{duration(vid):.3f}", "-movflags", "+faststart", str(out)])
        print(f"rendered {out} ({duration(out):.1f}s) in {time.time() - t:.0f}s")
    print(f"outputs in {work}")

if __name__ == "__main__": main()
