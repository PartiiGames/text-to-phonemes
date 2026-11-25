English text to phonemes (CMUdict-based G2P, Piper/espeak-compatible IPA) in pure Rust, by Partii (https://partii.games), the place to make games together.

Pure-Rust English G2P (grapheme to phoneme) for on-device TTS. It turns text into espeak-ng **en-us** style IPA,
the phoneme set Piper VITS voices are trained on, and into Piper's phoneme ids. No C libraries, no network, no
runtime data files: the CMUdict lexicon and the out-of-vocabulary model are embedded.

```rust
let g2p = text_to_phonemes::G2p::new();
let sentences = g2p.sentences("Hello, world. It costs $3.50!");   // one String per sentence, for streaming
assert_eq!(sentences[0], "həlˈoʊ, wˈɜːld.");

// the id map comes from a Piper voice config (<voice>.onnx.json); this one is a tiny stand-in
let cfg = r#"{"phoneme_id_map": {"_": [0], "^": [1], "$": [2], "h": [20], "ə": [59]}}"#;
let map = text_to_phonemes::phoneme_id_map_from_config(cfg).unwrap();
assert_eq!(text_to_phonemes::to_ids("hə", &map), [1, 0, 20, 0, 59, 0, 2]);   // ^ _ h _ ə _ $
```

Or from the command line: `cargo run --example phonemize -- "Hi there!" en_US-hifi_tts_92-medium.onnx.json`.

## API
- `G2p::new()`; `g2p.sentences(text) -> Vec<String>` (IPA per sentence); `g2p.word(word)` (one word said alone);
  `g2p.arpabet(word)` / `g2p.oov_arpabet(word)`; `g2p.normalize(text)` (sentences of `Tok` words, letters and
  punctuation) and `g2p.sentence_ipa(&toks)`.
- `phoneme_id_map_from_config(json)`, `to_ids(ipa, &map)` (Piper's `phonemes_to_ids`: `^`, `_`, then each
  phoneme's ids followed by `_`, then `$`; unknown phonemes skipped; NFD code points; a config's `vowel_clusters`
  merged first), `piper_phonemes`, `missing_phonemes`.

## How it works
1. **Normalization:** sentences and clause marks (`. , ! ? ; :`, kept the way Piper keeps espeak's terminators:
   `", "` inside a sentence, the final mark ends it), numbers (cardinals, decimals, negatives, years, decades,
   ordinals), money ($ £ €, `$5m`), percents, times (`3:30 pm`, `9:00`), abbreviations (Mr. Dr. St. etc. e.g.
   a.m. ...), symbols (& + = @ # / °), and unknown all-caps acronyms spelled out.
2. **Lexicon:** CMUdict (first pronunciation of each word), built by `build.rs` into a front-coded sorted table and
   looked up in place by binary search.
3. **Out of vocabulary:** a letter-context model trained only on CMUdict: Viterbi-EM letter-to-phone alignment
   (each letter emits 0-2 phones), then the most frequent output per letter window in a nested back-off chain from
   4+4 letters of context down to the letter alone, pruned to the windows that change the prediction.
4. **IPA:** ARPAbet with stress to espeak-ng en-us conventions: stress marks before the vowel, `ː` on ɑː ɔː iː uː
   ɜː, ɚ and ɜː(ɹ), ɐ, ᵻ, the t flap ɾ, ʔn̩, the ʲ glide; reduced, unstressed function words inside sentences.

## Licenses
- Code: MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`).
- Data: `data/cmudict.dict` is the CMU Pronouncing Dictionary (BSD-style license, `LICENSE-CMUDICT`; source and
  commit in `NOTICE`), and `data/oov.model` is trained only on it. Binaries must reproduce the CMUdict notice.
- **espeak-ng (GPL-3.0) is not a dependency** and is never linked or shipped. It appears only in the repository's
  farm test scripts (`tools/`, not in the published package), which run Piper's own phonemizer as an offline
  reference to measure agreement and tune the rules.

## Development (repository only)
`tools/farm.sh` builds, tests, (re)trains the OOV model (`RETRAIN=1`) and runs `tools/compare.py` against Piper's
espeak phonemizer (`piper-tts==1.8.0` in a venv): word and sentence agreement, top confusions, symbol coverage
against a voice's phoneme_id_map, and `to_ids` against Piper's `phonemes_to_ids`. `src/bin/g2p-tool.rs` has the
`train`, `words`, `sents`, `ids` and `stats` commands it uses.
