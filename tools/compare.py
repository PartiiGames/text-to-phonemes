"""Farm-only, test-only: compare text-to-phonemes's IPA with espeak-ng en-us through Piper's own phonemizer.

espeak-ng (GPL-3.0) comes from the piper-tts wheel and runs only here, as a reference. Nothing it prints is
copied into the crate; the report guides hand-tuning of the ARPAbet-to-IPA rules.

usage: python compare.py <g2p-tool> <words.txt> <nwords> <sentences.txt> <voice.onnx.json> <out dir>
"""

import collections
import json
import subprocess
import sys
import time
import unicodedata

from piper.phoneme_ids import phonemes_to_ids
from piper.phonemize_espeak import EspeakPhonemizer

STRESS = "ˈˌ"
MODS = "ːʲ̩"


def tokens(ipa, keep_stress=True):
    """IPA string to phoneme tokens: length/palatal/syllabic marks stay with their base, diphthongs and
    affricates are one token."""
    out = []
    for c in ipa:
        if c in MODS and out:
            out[-1] += c
        elif c in STRESS and not keep_stress:
            continue
        else:
            out.append(c)
    merged = []
    for t in out:
        if merged and (merged[-1] + t[0]) in ("aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "tʃ", "dʒ"):
            merged[-1] += t
        else:
            merged.append(t)
    return merged


def align(a, b):
    """Levenshtein alignment of token lists: (distance, [(ref, hyp)] with None for gaps)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, b[j - 1]))
            j -= 1
    return d[n][m], pairs[::-1]


def run_tool(tool, mode, lines):
    p = subprocess.run([tool, mode], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        sys.exit(f"g2p-tool {mode} failed: {p.stderr[-2000:]}")
    return p.stdout.rstrip("\n").split("\n")


def main():
    tool, words_path, nwords, sents_path, voice_path, out_dir = sys.argv[1:7]
    nwords = int(nwords)
    t0 = time.time()
    ph = EspeakPhonemizer()

    def espeak(text):
        return ["".join(s) for s in ph.phonemize("en-us", text)]

    # ---- words
    words = []
    for w in open(words_path, encoding="utf-8"):
        w = w.strip().lower()
        if w.isalpha() and w not in words:
            words.append(w)
        if len(words) >= nwords:
            break
    ours = dict(l.split("\t", 1) for l in run_tool(tool, "words", words))
    exact = exact_ns = 0
    err = err_ns = tot = 0
    conf = collections.Counter()
    bad = []
    for w in words:
        ref = unicodedata.normalize("NFC", "".join(espeak(w)).strip())
        hyp = ours.get(w, "")
        rt, ht = tokens(ref), tokens(hyp)
        d, pairs = align(rt, ht)
        dn, _ = align(tokens(ref, False), tokens(hyp, False))
        err += d
        err_ns += dn
        tot += len(rt)
        exact += d == 0
        exact_ns += dn == 0
        if d:
            bad.append((w, ref, hyp))
            for r, h in pairs:
                if r != h:
                    conf[(r or "-", h or "-")] += 1
    nw = len(words)
    print(f"== words: {nw} (most frequent first, {words_path})")
    print(f"word exact match {100 * exact / nw:.1f}%  ignoring stress marks {100 * exact_ns / nw:.1f}%")
    print(f"token error rate vs espeak {100 * err / tot:.2f}%  (no stress marks {100 * err_ns / tot:.2f}%)")
    print("top confusions (espeak -> ours):")
    for (r, h), c in conf.most_common(45):
        print(f"  {c:5d}  {r} -> {h}")
    print("first 60 mismatched words (word, espeak, ours):")
    for w, r, h in bad[:60]:
        print(f"  {w}\t{r}\t{h}")

    # ---- cross-check with the system espeak-ng CLI (same flags as the brief), when the node has it
    import shutil
    if shutil.which("espeak-ng"):
        ver = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True).stdout.strip()
        sample = words[:1000]
        p = subprocess.run(["espeak-ng", "-v", "en-us", "-q", "--ipa"], input="\n".join(sample) + "\n",
                           capture_output=True, text=True, timeout=600)
        cli = [l.strip() for l in p.stdout.split("\n") if l.strip()]
        if len(cli) == len(sample):
            ours_eq = sum(unicodedata.normalize("NFC", c) == ours.get(w, "") for w, c in zip(sample, cli))
            piper_eq = sum(unicodedata.normalize("NFC", c) == unicodedata.normalize("NFC", "".join(espeak(w)).strip())
                           for w, c in zip(sample, cli))
            print(f"system {ver}: first {len(sample)} words, ours == CLI {100 * ours_eq / len(sample):.1f}%, "
                  f"Piper phonemizer == CLI {100 * piper_eq / len(sample):.1f}%")
        else:
            print(f"system {ver}: CLI gave {len(cli)} lines for {len(sample)} words; skipped")
    else:
        print("system espeak-ng: not on this node (Piper's bundled espeak-ng is the reference)")

    # ---- sentences
    sents = [l.strip() for l in open(sents_path, encoding="utf-8") if l.strip()]
    ours_s = run_tool(tool, "sents", sents)
    s_exact = s_err = s_err_ns = s_tot = 0
    num_exact = num_n = 0
    wconf = collections.Counter()
    sconf = collections.Counter()
    shown = []
    all_ours = []
    for text, hyp_line in zip(sents, ours_s):
        ref = unicodedata.normalize("NFC", " ".join(espeak(text)))
        hyp = " ".join(hyp_line.split("\t"))
        all_ours.append(hyp)
        rt, ht = tokens(ref), tokens(hyp)
        d, pairs = align(rt, ht)
        dn, _ = align([t for t in tokens(ref, False) if t != " "], [t for t in tokens(hyp, False) if t != " "])
        s_err += d
        s_err_ns += dn
        s_tot += len(rt)
        s_exact += d == 0
        has_num = any(c.isdigit() for c in text) or "." in text[:-1]
        if not has_num:
            num_n += 1
            num_exact += d == 0
        for r, h in pairs:
            if r != h:
                sconf[(r or "-", h or "-")] += 1
        rw, hw = ref.split(" "), hyp.split(" ")
        if len(rw) == len(hw):
            for a, b in zip(rw, hw):
                if a != b:
                    wconf[(a, b)] += 1
        if d and len(shown) < 25:
            shown.append((text, ref, hyp))
    ns = len(sents)
    print(f"\n== sentences: {ns}")
    print(f"sentence exact match {100 * s_exact / ns:.1f}%  (without digits/abbreviations: {num_exact}/{num_n})")
    print(f"token error rate vs espeak {100 * s_err / s_tot:.2f}%  (no stress marks or spaces {100 * s_err_ns / s_tot:.2f}%)")
    print("top token confusions in sentences (espeak -> ours):")
    for (r, h), c in sconf.most_common(30):
        print(f"  {c:5d}  {r!r} -> {h!r}")
    print("top word confusions in sentences (espeak -> ours):")
    for (r, h), c in wconf.most_common(60):
        print(f"  {c:4d}  {r}  ->  {h}")
    print("mismatched sentences:")
    for t, r, h in shown:
        print(f"  {t}\n    espeak {r}\n    ours   {h}")

    # ---- symbols and ids
    cfg = json.load(open(voice_path, encoding="utf-8"))
    idmap = cfg["phoneme_id_map"]
    print(f"\n== voice {voice_path}: {len(idmap)} ids, vowel_clusters={cfg.get('vowel_clusters')}, keys={sorted(cfg)}")
    used = collections.Counter(c for s in all_ours + list(ours.values()) for c in unicodedata.normalize("NFD", s))
    missing = {c: n for c, n in used.items() if c not in idmap}
    print(f"symbols we emit: {len(used)}: {''.join(sorted(used))}")
    print(f"missing from the voice's phoneme_id_map: {missing or 'none'}")
    # our to_ids vs piper's phonemes_to_ids on the same phoneme strings
    p = subprocess.run([tool, "ids", voice_path], input="\n".join(all_ours) + "\n", capture_output=True, text=True, timeout=600)
    ours_ids = [json.loads(l) for l in p.stdout.strip().split("\n")]
    clusters = {tuple(vc) for vc in cfg.get("vowel_clusters") or []}
    same = 0
    for s, ids in zip(all_ours, ours_ids):
        phs = list(unicodedata.normalize("NFD", s))
        if clusters:
            from piper.phonemize_espeak import _merge_known_vowel_clusters
            phs = _merge_known_vowel_clusters(phs, clusters)
        same += phonemes_to_ids(phs, idmap) == ids
    print(f"to_ids equal to piper's phonemes_to_ids: {same}/{len(all_ours)}")

    summary = {
        "words": nw, "word_exact": exact / nw, "word_exact_no_stress": exact_ns / nw, "word_ter": err / tot,
        "word_ter_no_stress": err_ns / tot, "sentences": ns, "sentence_exact": s_exact / ns,
        "sentence_ter": s_err / s_tot, "sentence_ter_no_stress_space": s_err_ns / s_tot,
        "missing_symbols": missing, "ids_equal": f"{same}/{len(all_ours)}", "seconds": time.time() - t0,
    }
    json.dump(summary, open(f"{out_dir}/compare.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
