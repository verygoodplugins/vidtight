#!/usr/bin/env python3
"""vidtight: local screen-recording tightener. ASR (parakeet-mlx) -> fillers/pauses/retakes -> vocab fix -> EDL -> SRT + ffmpeg render."""
import argparse, difflib, json, re, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
import yaml

FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm", "umm"}
SOFT_START = {"so", "like", "basically", "okay", "ok", "right"}  # filler only at sentence start
PHRASES = [("you", "know"), ("kind", "of"), ("sort", "of")]  # ("i","mean") opt-in via --cut-i-mean

LEAD = 0.0  # seconds of title card prepended (set by --leadin)

def sh(cmd): return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

def duration(p): return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))

def norm(w): return re.sub(r"[^\w']", "", w.lower())

# (a)+(b) audio + local ASR with word timestamps
def transcribe(src, work, model):
    wav = work / "audio.wav"
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)])
    from parakeet_mlx import from_pretrained
    t = time.time(); m = from_pretrained(model)
    res = m.transcribe(str(wav), chunk_duration=120, overlap_duration=15)
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

def snap(keep, rms, win):
    """Move each cut edge to the quietest frame within win (keep-end looks back, keep-start looks forward)."""
    if win <= 0: return keep
    end = len(rms) / 100
    out = [[0.0 if s < 0.01 else quiet(rms, s - 0.03, s + win), e if e >= end - 0.01 else quiet(rms, e - win, e + 0.03), sp] for s, e, sp in keep]
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
    p.add_argument("src"); p.add_argument("--vocab", default=str(Path(__file__).with_name("vocab.yaml")))
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
    a = p.parse_args()
    global LEAD
    if a.leadin: LEAD = duration(a.leadin)
    src = Path(a.src).expanduser().resolve(); work = Path(a.out).expanduser() if a.out else src.parent / (src.stem + ".vidtight"); work.mkdir(parents=True, exist_ok=True)
    if a.cut_i_mean: PHRASES.append(("i", "mean"))
    a.dur = duration(src); a.src = src; a.wav = work / "audio.wav"
    fps = round(Fraction(sh(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(src)]).strip()))
    ed = yaml.safe_load(open(a.edits)) if a.edits else {}
    tj = work / "words.json"
    if a.reuse and tj.exists(): words = json.loads(tj.read_text())
    else:
        words, asr_t = transcribe(src, work, a.model); tj.write_text(json.dumps(words))
        print(f"ASR: {a.dur:.1f}s audio in {asr_t:.1f}s ({a.dur / asr_t:.0f}x realtime)")
    fixes = apply_vocab(words, yaml.safe_load(open(a.vocab)) or {})
    rms = load_rms(a.wav)
    cuts = fix_fillers(find_cuts(words, a), rms)
    # manual edits
    keepr = ed.get("keep_retakes", [])
    dropped = [c for c in cuts if c[2] == "retake" and any(abs(c[0] - t) < 2 for t in keepr)]
    for c in dropped:
        for w in words:
            if c[0] <= w["s"] < c[1]: w["cut"] = False
    cuts = [c for c in cuts if c not in dropped]
    speed = []
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
    name = a.name or src.stem
    srt = work / f"{name}.srt"
    write_srt(words, keep, srt, ed.get("replace"), extra)
    (work / "edl.json").write_text(json.dumps({"src": str(src), "keep": keep, "cuts": cuts, "speed": speed, "vocab_fixes": fixes}, indent=1))
    with open(work / "cuts.csv", "w") as f:
        f.write("start,end,type,detail\n")
        for s, e, k, d in sorted(cuts): f.write(f"{ts(s, '.')},{ts(e, '.')},{k},\"{d}\"\n")
        for s, e, sp in speed: f.write(f"{ts(s, '.')},{ts(e, '.')},speed x{sp},\"{e - s:.1f}s -> {(e - s) / sp:.1f}s\"\n")
    by = {}
    for c in cuts: by[c[2]] = by.get(c[2], 0) + 1
    print(f"cuts: {by} speed-ups: {len(speed)} vocab fixes: {len(fixes)}")
    print(f"duration: {a.dur:.1f}s -> {outdur:.1f}s edit (+{tail:.1f}s music tail) = {outdur + tail:.1f}s (+{LEAD:.1f}s lead-in), {len(keep)} segments")
    if a.render:
        t = time.time()
        aud, vid, out = work / f"{name}.audio.m4a", work / f"{name}.video.mp4", work / f"{name}.mp4"
        render_audio(src, keep, aud, music); print(f"audio done {time.time() - t:.0f}s", flush=True)
        render_video(src, keep, vid, fps, a.bitrate, tail, fade=music.get("fade", tail) if music else 0.0, leadin=a.leadin, fadein=a.fadein)
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-i", str(aud), "-i", str(srt), "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-shortest", "-movflags", "+faststart", str(out)])
        print(f"rendered {out} ({duration(out):.1f}s) in {time.time() - t:.0f}s")
    print(f"outputs in {work}")

if __name__ == "__main__": main()
