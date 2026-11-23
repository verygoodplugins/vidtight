#!/usr/bin/env python3
"""vidtight: local screen-recording tightener. ASR (parakeet-mlx) -> fillers/pauses/retakes -> vocab fix -> EDL -> SRT + ffmpeg render.
--select turns it round: keep only the windows around wanted moments, with every edge in a speech gap.
--broll keeps the speech-free windows instead of cutting them as pauses."""
import argparse, csv, difflib, json, math, re, subprocess, sys, tempfile, time
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
JOIN = 0.5  # clips closer together than this are merged into one (twice --air when that is larger)

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
    return len(items)  # 0 when the cut has no speech and no captions; the mux must not require a subtitle stream then

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

def speech_gaps(blocks, dur, gap=0.4, mingap=0.15, tail=0.15, lead=MAXPAD, floor=MINPAD, air=0.0):
    """The gap before each block and after the last, each {s, e, prev, next, grade, lo, hi}. grade is "clean" (gap s or more, or open to the start or
    end of the file), "tight" (mingap s or more, and only between two caller intervals: ASR word times drift by more than a tight gap is wide, so a
    short gap seen only by the ASR cannot be trusted) or None (not cuttable). An edge may sit anywhere in [lo, hi]: tail s after the word before the
    gap and lead s before the word after it (ASR can stamp a start late). A narrower gap gets one point, the pads scaled to fit, neither under floor.
    air asks for breathing room: every edge stays air s clear of speech where the pause is long enough and sits mid-pause where it is not, and
    tight gaps are off, since they have no room to give."""
    out, end, prev = [], 0.0, None
    for b in [*blocks, None]:
        s = max(end, dur if b is None else b["s"]); w = s - end; a = min(air, w / 2)
        if prev is None or b is None: grade, lo, hi = "clean", (0.0 if prev is None else min(s, end + min(max(tail, air), w))), (s if b is None else max(end, s - min(max(lead, air), w)))
        else:
            grade = "clean" if w >= gap - 1e-9 else "tight" if air <= 0 and w >= mingap - 1e-9 and prev["src"] == b["src"] == "app" else None
            lo, hi = end + max(tail, a), s - max(lead, a)
            if lo > hi + 1e-9: lo = hi = end + min(max(w * tail / (tail + lead), floor), w - floor)
            else: hi = max(lo, hi)
        out.append({"s": end, "e": s, "prev": prev, "next": b, "grade": grade, "lo": lo, "hi": hi})
        if b and b["e"] >= end: end, prev = b["e"], b
    return out

def select_moments(moments, gaps, blocks, dur, pre=1.0, post=4.0, maxclip=12.0, join=JOIN):
    """A keep-window around each moment {t, label, pre?, post?, caption?, section?, note?} with both edges in a usable speech gap. A requested edge that lands in speech
    moves to a gap: outward takes the utterance whole (the clip may grow to maxclip), inward trims the window (never below KEEPMIN of it). The cheapest
    pair of moves wins, a tight gap costing TIGHT s extra; when no pair fits the moment is dropped. Returns (clips, rows): clips are [in, out] in source
    order, overlapping ones and ones less than join s apart merged; rows, one per moment in the caller's order, say what was done and why."""
    use = [g for g in gaps if g["grade"]]
    said = lambda x0, x1: " ".join(b["w"] for b in blocks if x0 <= (b["s"] + b["e"]) / 2 <= x1)  # speech a clip contains
    hit = lambda x0, x1: shorten(" ".join(b["w"] for b in blocks if b["s"] < x1 and x0 < b["e"]), 44)  # speech an edge moved across
    rows = []
    for m in moments:
        t = float(m["t"]); a0, b0 = max(0.0, t - float(m.get("pre", pre))), min(dur, t + float(m.get("post", post)))
        r = {"label": str(m.get("label", "")), "t": t, "want": [a0, b0], "status": "dropped", "clip": None, "in": None, "out": None,
             "note": "", "speech": "", "caption": m.get("caption"), "section": str(m.get("section") or ""), "shot_note": str(m.get("note") or "")}
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
        if clips and r["in"] <= clips[-1][1] + join: clips[-1][1] = max(clips[-1][1], r["out"])
        else: clips.append([r["in"], r["out"]])
        r["clip"] = len(clips)
    for r in rows:
        if r["clip"]:
            r["in"], r["out"] = clips[r["clip"] - 1]; r["speech"] = said(r["in"], r["out"])
            if sum(x["clip"] == r["clip"] for x in rows) > 1: r["status"] += "+merged"
    return clips, rows

def windows_of(gaps, minlen):
    """Speech-free windows from gaps already built by speech_gaps. A file with no speech at all is one window, the whole file, however short:
    that is the silent hero shot pause-cutting would otherwise eat. Any other gap under minlen is a breath inside chatter, not a shot.
    lo/hi are the padded edges a cut may sit on; start/end are the speech-free span itself."""
    out = []
    for g in gaps:
        d = g["e"] - g["s"]
        if d <= 1e-9: continue
        if not (g["prev"] is None and g["next"] is None) and d < minlen - 1e-9: continue
        out.append({"start": g["s"], "end": g["e"], "duration": d, "lo": g["lo"], "hi": g["hi"]})
    return out

def broll_windows(blocks, dur, minlen=0.8, **kw):
    """Speech-free windows {start, end, duration, lo, hi}. kw is forwarded to speech_gaps (gap, air, the pads)."""
    return windows_of(speech_gaps(blocks, dur, **kw), minlen)

def select_broll(moments, windows, blocks, dur):
    """Keep the speech-free windows instead of cutting them. With moments, each one keeps the whole window containing t, so a marked silent shot
    stays whole instead of shrinking to a pre/post slice. With none, every window is a clip and setup chatter is left out. Rows match select_moments;
    section and note are the caller's, and note on the row stays the reason an edge moved."""
    said = lambda x0, x1: " ".join(b["w"] for b in blocks if x0 <= (b["s"] + b["e"]) / 2 <= x1)  # speech a clip contains
    usable = [w for w in windows if w["hi"] - w["lo"] > 0.05]
    if not moments: return [[w["lo"], w["hi"]] for w in usable], []
    rows = []
    for m in moments:
        t = float(m["t"])
        r = {"label": str(m.get("label", "")), "t": t, "want": None, "status": "dropped", "clip": None, "in": None, "out": None,
             "note": "", "speech": "", "caption": m.get("caption"), "section": str(m.get("section") or ""), "shot_note": str(m.get("note") or "")}
        rows.append(r)
        if not 0 <= t <= dur: r["note"] = "t is outside the recording"; continue
        w = next((w for w in usable if w["start"] - 1e-9 <= t <= w["end"] + 1e-9), None)
        if not w: r["note"] = "t is inside speech, or the silent gap there is shorter than --pause"; continue
        r.update({"in": w["lo"], "out": w["hi"], "want": [w["lo"], w["hi"]], "status": "kept"})
    clips = []
    for r in sorted((r for r in rows if r["in"] is not None), key=lambda r: r["in"]):
        if clips and r["in"] <= clips[-1][1] + 1e-6: clips[-1][1] = max(clips[-1][1], r["out"])
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
    because the ASR stamped that word as still running at t. Only the exact start and end of the file are exempt: an edge a few frames short of the
    end still cuts a line that runs to the end."""
    out = []
    for s, e, *_ in keep:
        for side, x in (("in", s), ("out", e)):
            d = {"t": x, "side": side, "safe": True, "inside": False, "tight": False, "est": False, "why": "file start" if x <= 0.001 else "file end"}
            out.append(d)
            if x <= 0.001 or x >= dur - 0.001: continue
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

def settle(clips, gaps, rms, win, fps, dur):
    """Selected clips -> final keep-segments: snap each edge inside its own gap (pads intact), put it on the frame grid, and leave an edge at the very
    end of the file there, because rounding it to the grid could pull it back into a last word that runs to the end."""
    lim = lambda t: next(((g["lo"], g["hi"]) for g in gaps if g["s"] <= t <= g["e"]), (t, t))
    keep = quantize(snap([[i, o, 1] for i, o in clips], rms, win, lim), fps)
    if len(keep) != len(clips): sys.exit("vidtight: a selected clip collapsed while snapping; rerun with --snap 0")
    for k, c in zip(keep, clips):
        if c[1] >= dur - 1e-6: k[1] = dur
    return keep

def select_edl(a, raw, words, rms, fps):
    """--select / --broll: keep-segments around the wanted moments, or the speech-free windows. Returns (keep, cuts, sel); sel carries the per-moment
    rows, the speech map and caption words. The speech map is built from raw, the words as transcribed: vocabulary fixes merge several words into one
    span, which the per-word length cap would then cut short. words (vocabulary applied) only feed the captions."""
    moments = []
    if a.select:
        y = yaml.safe_load(Path(a.select).expanduser().read_text()) or []
        moments = y.get("moments", []) if isinstance(y, dict) else y
    speech = (yaml.safe_load(Path(a.speech).expanduser().read_text()) or []) if a.speech else []
    sil = [(s0, a.dur if e0 is None else e0) for s0, e0 in silences(a.wav, a.noise, 0.3)]
    blocks = speech_blocks(raw, speech, sil, a.max_word); hf = 0.5 / fps  # half a frame of slack so frame quantising cannot eat a pad
    gaps = speech_gaps(blocks, a.dur, a.gap, max(a.min_gap, 0.12), min(max(a.pad, MINPAD), MAXPAD) + hf, MAXPAD, MINPAD + hf, max(a.air, 0.0))
    windows = windows_of(gaps, a.pause) if a.broll else None
    if a.broll: clips, rows = select_broll(moments, windows, blocks, a.dur)
    else: clips, rows = select_moments(moments, gaps, blocks, a.dur, a.pre, a.post, a.max_clip, max(JOIN, 2 * a.air))
    keep = settle(clips, gaps, rms, a.snap, fps, a.dur)
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
    asked = check_edl([[*r["want"], 1] for r in rows if r.get("want") and 0 <= r["t"] <= a.dur], blocks, a.dur, a.gap)
    capw = [{**w, "e": min(w["e"], w["s"] + wordcap(w["w"], a.max_word) * len(w["w"].split()))} for w in words]  # captions must not ride a smeared word into another clip
    return keep, cuts, {"rows": rows, "blocks": blocks, "utts": utterances(blocks, a.gap), "words": capw, "asked": asked, "sil": len(sil), "windows": windows}

def window_rows(windows):
    """The windows.json / edl shape: start, end, duration. The padded lo/hi stay internal; they are where a cut edge sits, not the speech-free span."""
    rows = []
    for w in windows:
        s, e = round(float(w["start"]), 3), round(float(w["end"]), 3)
        rows.append({"start": s, "end": e, "duration": round(e - s, 3)})
    return rows

def write_windows(work, windows):
    rows = window_rows(windows)
    (Path(work) / "windows.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"b-roll: {len(rows)} speech-free window(s), {sum(w['duration'] for w in rows):.1f}s")
    for w in rows: print(f"  {w['start']:.2f}-{w['end']:.2f} ({w['duration']:.2f}s)")
    return rows

def write_storyboard(rows, work):
    """storyboard.md and shots.json beside select.csv: the shot list to read before a render. section groups the markdown; note is the caller's."""
    shots = []
    for i, r in enumerate(rows, 1):
        clip = r["clip"]
        shots.append({"n": i, "t": round(float(r["t"]), 3), "label": r["label"], "section": r.get("section") or "", "note": r.get("shot_note") or "",
                      "status": r["status"], "clip": clip, "in": None if not clip else round(float(r["in"]), 3), "out": None if not clip else round(float(r["out"]), 3),
                      "duration": None if not clip else round(float(r["out"]) - float(r["in"]), 3), "speech": r["speech"], "caption": r.get("caption") or "", "edge": r["note"]})
    (Path(work) / "shots.json").write_text(json.dumps(shots, indent=1) + "\n")
    lines, groups = ["# Storyboard", ""], []
    for s in shots:
        if not groups or groups[-1][0] != s["section"]: groups.append((s["section"], []))
        groups[-1][1].append(s)
    for sec, group in groups:
        if sec: lines += [f"## {sec}", ""]
        for s in group:
            span = f"{s['in']:.2f}-{s['out']:.2f} ({s['duration']:.2f}s), clip {s['clip']}, {s['status']}" if s["clip"] else s["status"]
            lines.append(f"{s['n']}. **{s['label'] or 'untitled'}** — {s['t']:.2f}s, {span}")
            if s["note"]: lines.append(f"   {s['note']}")
            if s["edge"]: lines.append(f"   {s['edge']}")
            if s["speech"]: lines.append(f"   speech: {s['speech']}")
            lines.append("")
    (Path(work) / "storyboard.md").write_text("\n".join(lines).rstrip() + "\n")

def report_select(a, work, sel):
    """Write select.csv, storyboard.md and shots.json, say what happened to each moment, then re-read the EDL against the speech map. Returns unsafe edges."""
    rows, blocks = sel["rows"], sel["blocks"]
    with open(work / "select.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["moment", "t", "status", "clip", "in", "out", "dur", "in_safe", "in_why", "out_safe", "out_why", "note", "speech", "caption", "section", "shot_note"])
        for r in rows:
            c = r["clip"]; yn = lambda k: ("yes" if r[k] else "no") if c else ""
            w.writerow([r["label"], f"{r['t']:.3f}", r["status"], c or "", *([f"{r['in']:.3f}", f"{r['out']:.3f}", f"{r['out'] - r['in']:.3f}"] if c else [""] * 3),
                        yn("in_safe"), r.get("in_why", ""), yn("out_safe"), r.get("out_why", ""), r["note"], r["speech"], r["caption"] or "", r.get("section", ""), r.get("shot_note", "")])
    write_storyboard(rows, work)
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

def report_broll(a, work, sel):
    """windows.json, then the same edge self-check as --select. There are no moment rows: the windows are the cut."""
    write_windows(work, sel["windows"])
    blocks = sel["blocks"]
    est = [b for b in blocks if b["src"] == "cap"]
    if est: print(f"speech map: LOW CONFIDENCE. {len(est)} word end(s) are estimates; a window edge after one of them is inferred, not measured.")
    edges = check_edl(json.loads((work / "edl.json").read_text())["keep"], blocks, a.dur, a.gap)
    bad = [e for e in edges if not e["safe"]]
    print(f"self-check: {len(edges)} edges re-read from edl.json: {sum(e['inside'] for e in edges)} inside speech, {len(bad)} not speech-safe")
    if bad:
        print(f"WARNING: {len(bad)} EDGE(S) NOT SPEECH-SAFE. Do not ship this cut.", file=sys.stderr)
        for e in bad: print(f"  {e['side']} edge at {e['t']:.3f}s: {e['why']}", file=sys.stderr)
    return bad

# render: audio in one pass (fades, music, loudnorm, true-peak guard), video in chunks (hw encode), then mux
TP = -1.0  # dBTP ceiling of the encoded master. AAC still peaks above a sample limiter, so the file is measured and the audio re-encoded until it is at or under this.
LIMIT = 0.83  # starting alimiter ceiling (linear, level=false). 4x oversampled, 0.83 still measured about -0.5 dBTP; about 0.76 cleared -1 at the same LUFS.
OSRATE = 192000  # 4x oversampling for the limiter; delivery stays 48 kHz

def limiter_chain(limit):
    """loudnorm, then a 4x-oversampled alimiter. level=false holds the integrated loudness; only the ceiling moves when the encoded peak is hot."""
    return f"loudnorm=I=-12.5:TP=-1.5:LRA=11,aresample={OSRATE},alimiter=limit={limit:.4f}:level=false,aresample=48000"

def parse_loudness(err):
    """(true peak dBTP, integrated LUFS) from an ebur128 summary, else a loudnorm JSON print. None when neither is there."""
    if err and "Summary:" in err:
        tail = err.rsplit("Summary:", 1)[1]
        i, p = re.search(r"I:\s+([+-]?[\d.]+) LUFS", tail), re.search(r"Peak:\s+([+-]?[\d.]+) dBFS", tail)
        if i and p: return float(p.group(1)), float(i.group(1))
    m = re.search(r"\{[^{}]*\"input_tp\"[^{}]*\}", err or "", re.S)
    if not m: return None
    d = json.loads(m.group(0))
    return float(d["input_tp"]), float(d["input_i"])

def measure_loudness(path):
    """True peak and integrated LUFS of an encoded file. The meter sees the decoded samples, so the peak is the codec's, not the limiter's."""
    path = str(Path(path).resolve())
    def run(af): return subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", path, "-af", af, "-f", "null", "-"], capture_output=True, text=True).stderr
    got = parse_loudness(run("ebur128=peak=true:framelog=quiet")) or parse_loudness(run("loudnorm=print_format=json"))
    if not got: sys.exit(f"vidtight: could not read the true peak of {path}")
    return got

def guard_true_peak(encode, measure, limit=LIMIT, ceiling=TP, tries=5):
    """encode(limit) writes the audio; measure() -> (tp, lufs) of that encoded file. Lowers the ceiling and re-encodes the audio only until tp <= ceiling.
    Returns (limit, tp, lufs). The video is not touched. A pass that cannot get there stops at the floor and says so."""
    for n in range(tries):
        encode(limit)
        tp, lufs = measure()
        if tp <= ceiling:
            print(f"loudness: {lufs:.1f} LUFS, {tp:.1f} dBTP")
            return limit, tp, lufs
        nxt = max(0.5, limit * 10 ** (-(tp - ceiling + 0.2) / 20))  # the overshoot, plus a little for the codec's intersample peak
        if n == tries - 1 or nxt >= limit - 1e-4:
            print(f"WARNING: encoded audio is {tp:.1f} dBTP (limiter {limit:.3f}); {ceiling:.0f} dBTP was not reached.", file=sys.stderr)
            print(f"loudness: {lufs:.1f} LUFS, {tp:.1f} dBTP")
            return limit, tp, lufs
        print(f"loudness: {tp:.1f} dBTP is over {ceiling:.0f}, re-encoding audio at limiter {nxt:.3f}")
        limit = nxt
    raise RuntimeError("guard_true_peak needs at least one try")  # the loop returns; this is only reached if tries < 1

def render_audio(src, keep, out, music=None, xf=0.02):
    """Assemble the voice edit in numpy (fast, sample-exact), then ffmpeg for music, loudnorm and a true-peak-guarded limiter."""
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
    def encode(limit):
        inputs, fg = ["-i", str(voice)], ""
        if music:
            inputs += ["-i", str(music["file"])]
            st, slam, duck = music["start_out"] + LEAD, music["slam_out"] + LEAD, 10 ** (music.get("duck_db", -19) / 20)
            fg = (f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                  f"volume='{duck}+(1-{duck})*clip((t-{slam - st:.3f})/0.5,0,1)':eval=frame,afade=t=out:st={music['dur'] - music.get('afade', 1.5):.3f}:d={music.get('afade', 1.5):.3f},"
                  f"adelay={int(st * 1000)}:all=1[mus];[0:a]apad=whole_dur={st + music['dur']:.3f}[vp];"
                  f"[vp][mus]amix=inputs=2:normalize=0:duration=longest,")
        else: fg = "[0:a]"
        fade = f",afade=t=out:st={music['start_out'] + LEAD + music['dur'] - 2.5:.3f}:d=2.5" if music else ""
        fg += limiter_chain(limit) + fade + "[aout]"
        sh(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fg, "-map", "[aout]", "-c:a", "aac_at", "-b:a", "256k", str(out)])
    guard_true_peak(encode, lambda: measure_loudness(out))

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

def contact_times(dur, every):
    """Source times of one frame every `every` seconds, from 0 up to but not past the end."""
    if every <= 0: raise ValueError("contact interval must be positive")
    out, i = [], 0
    while i * every < dur - 1e-3:
        out.append(i * every); i += 1
    return out or [0.0]

def tile_contact(frames, dest, cell_w=320):
    """Label each frame with its timecode and tile them into one PNG. Every frame is scaled to the same width and padded to the
    tallest cell before tiling, so mixed sizes still line up. Labels are drawn with Pillow: Homebrew ffmpeg often has no drawtext."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        sys.exit("vidtight: --contact needs Pillow (pip install pillow)")
    font = None
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            try: font = ImageFont.truetype(p, 16); break
            except OSError: font = None
    font = font or ImageFont.load_default()
    fitted = []
    for path, label in frames:
        im = Image.open(path).convert("RGB")
        fitted.append((im.resize((cell_w, max(1, round(im.height * cell_w / im.width)))), label))
    cell_h, bar, cards = max(im.height for im, _ in fitted), 24, []
    for im, label in fitted:
        canvas = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
        canvas.paste(im, (0, (cell_h - im.height) // 2))
        card = Image.new("RGB", (cell_w, cell_h + bar), (24, 18, 12))
        card.paste(canvas, (0, 0))
        ImageDraw.Draw(card).text((6, cell_h + 3), label, fill=(255, 214, 140), font=font)
        cards.append(card)
    n, cols = len(cards), 1
    while cols * cols < n: cols += 1
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + bar)), (0, 0, 0))
    for i, card in enumerate(cards): sheet.paste(card, ((i % cols) * cell_w, (i // cols) * (cell_h + bar)))
    dest = Path(dest).resolve(); dest.parent.mkdir(parents=True, exist_ok=True); sheet.save(dest)

def contact_sheet(src, dest, every):
    """A frame of src every `every` seconds, labelled with its source timecode and tiled into dest. ffmpeg only extracts frames; every path is absolute."""
    times, src = contact_times(duration(src), every), Path(src).resolve()
    with tempfile.TemporaryDirectory() as td:
        frames = []
        for i, t in enumerate(times):
            fp = (Path(td) / f"f{i:04d}.png").resolve()
            sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1", str(fp)])
            frames.append((fp, ts(t, ".")))
        tile_contact(frames, dest)
    print(f"contact sheet: {Path(dest).resolve()} ({len(times)} frames, every {every:g}s)")

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
    p.add_argument("--chunk", type=float, help="ASR chunk length s (default 120; 30 with --select or --broll, because long chunks drop lines when music runs under the voice)")
    p.add_argument("--select", help="moments.yaml [{t, label, pre?, post?, caption?, section?, note?}]: keep the windows around these moments, every edge in a speech gap")
    p.add_argument("--broll", action="store_true", help="keep speech-free windows of at least --pause seconds (windows.json) instead of cutting pauses; with --select, a moment keeps the silent window it falls in")
    p.add_argument("--contact", type=float, metavar="SECONDS", help="write a labelled contact sheet, one frame every N seconds, beside the analysis outputs")
    p.add_argument("--speech", help="--select/--broll: exact speech intervals [[start, end, label], ...] from the app; they replace the ASR words they overlap")
    p.add_argument("--pre", type=float, default=1.0, help="--select: seconds kept before each moment"); p.add_argument("--post", type=float, default=4.0, help="--select: seconds kept after each moment")
    p.add_argument("--max-clip", type=float, default=12.0, help="--select: longest a clip may grow to take an utterance whole (s)")
    p.add_argument("--gap", type=float, default=0.4, help="--select: smallest speech gap an edge may use (s)")
    p.add_argument("--min-gap", type=float, default=0.15, help="--select: smallest gap usable between two exact --speech intervals (s)")
    p.add_argument("--air", type=float, default=0.0, help="--select: voice-free footage kept at every edge next to speech, where the pause is long enough (s). Clips less than twice this apart are joined")
    p.add_argument("--max-word", type=float, default=1.2, help="--select: longest a single word can last; longer ASR stamps are cut back (s)")
    p.add_argument("--strict", action="store_true", help="--select: exit 2 if any edge of the final EDL is not speech-safe")
    a = p.parse_args()
    if a.speech and not (a.select or a.broll): p.error("--speech needs --select or --broll")
    if a.contact is not None and a.contact <= 0: p.error("--contact interval must be positive")
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
        chunk = a.chunk or (30 if a.select or a.broll else 120)
        words, asr_t = transcribe(src, work, a.model, *([chunk] if chunk != 120 else [])); tj.write_text(json.dumps(words))
        print(f"ASR: {a.dur:.1f}s audio in {asr_t:.1f}s ({a.dur / asr_t:.0f}x realtime)")
    if not a.wav.exists(): wav16k(src, a.wav)  # --words or --reuse into a fresh --out dir
    vf = Path(a.vocab).expanduser() if a.vocab else Path(__file__).with_name("vocab.yaml")  # the bundled default is absent in a non-editable install
    raw = [dict(w) for w in words]  # --select maps speech from the stamps as transcribed, before vocabulary fixes merge words
    fixes = apply_vocab(words, (yaml.safe_load(vf.read_text()) or {}) if a.vocab or vf.exists() else {})
    rms = load_rms(a.wav)
    speed, sel = [], None
    if a.select or a.broll: keep, cuts, sel = select_edl(a, raw, words, rms, fps)
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
    cues = write_srt(sel["words"] if sel else words, keep, srt, ed.get("replace"), extra)
    edl = {"src": str(src), "keep": keep, "cuts": cuts, "speed": speed, "vocab_fixes": fixes}
    if sel:
        edl["speech"] = sel["utts"]
        if a.select: edl["select"] = sel["rows"]
        if sel.get("windows") is not None: edl["windows"] = window_rows(sel["windows"])
    (work / "edl.json").write_text(json.dumps(edl, indent=1))
    with open(work / "cuts.csv", "w") as f:
        f.write("start,end,type,detail\n")
        for s, e, k, d in sorted(cuts): f.write(f"{ts(s, '.')},{ts(e, '.')},{k},\"{d}\"\n")
        for s, e, sp in speed: f.write(f"{ts(s, '.')},{ts(e, '.')},speed x{sp},\"{e - s:.1f}s -> {(e - s) / sp:.1f}s\"\n")
    by = {}
    for c in cuts: by[c[2]] = by.get(c[2], 0) + 1
    if sel and a.select:
        if a.broll: write_windows(work, sel["windows"])
        unsafe = report_select(a, work, sel)
    elif sel: unsafe = report_broll(a, work, sel)
    else: unsafe = []
    if not sel: print(f"cuts: {by} speed-ups: {len(speed)} vocab fixes: {len(fixes)}")
    print(f"duration: {a.dur:.1f}s -> {outdur:.1f}s edit (+{tail:.1f}s music tail) = {outdur + tail:.1f}s (+{LEAD:.1f}s lead-in), {len(keep)} segments")
    if a.contact: contact_sheet(src, work / "contact.png", a.contact)
    if unsafe and a.strict: print(f"outputs in {work}"); sys.exit(2)
    if a.render and not keep:
        why = f"no moment survived the selection (see {work / 'select.csv'})" if a.select else f"no speech-free window to keep (see {work / 'windows.json'})" if a.broll else "every second was cut"
        sys.exit("vidtight: nothing to render: " + why)
    if a.render:
        t = time.time()
        aud, vid, out = work / f"{name}.audio.m4a", work / f"{name}.video.mp4", work / f"{name}.mp4"
        render_audio(src, keep, aud, music); print(f"audio done {time.time() - t:.0f}s", flush=True)
        render_video(src, keep, vid, fps, a.bitrate, tail, fade=music.get("fade", tail) if music else 0.0, leadin=a.leadin, fadein=a.fadein)
        # end with the video, not -shortest: the caption track stops at the last spoken word, which would cut off any footage after it.
        # an empty SRT (a silent --broll window, no moment or music caption) is not an input: mapping a subtitle stream that is not there fails the mux
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-i", str(aud)]
        if cues: cmd += ["-i", str(srt)]
        cmd += ["-map", "0:v", "-map", "1:a"]
        if cues: cmd += ["-map", "2:s"]
        cmd += ["-c:v", "copy", "-c:a", "copy"]
        if cues: cmd += ["-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
        cmd += ["-t", f"{duration(vid):.3f}", "-movflags", "+faststart", str(out)]
        sh(cmd)
        print(f"rendered {out} ({duration(out):.1f}s) in {time.time() - t:.0f}s")
    print(f"outputs in {work}")

if __name__ == "__main__": main()
