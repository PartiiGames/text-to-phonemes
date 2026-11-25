//! # text-to-phonemes
//!
//! English text to phonemes for on-device TTS, in pure Rust. The output is espeak-ng **en-us** style IPA, the
//! phoneme set Piper VITS voices are trained on, plus Piper's phoneme ids.
//!
//! ```
//! let g2p = text_to_phonemes::G2p::new();
//! let sentences = g2p.sentences("Hello, world. It costs $3.50!");
//! assert_eq!(sentences.len(), 2);
//! assert_eq!(sentences[0], "həlˈoʊ, wˈɜːld.");
//!
//! // a voice config (`<voice>.onnx.json`) gives the id map; this one is a tiny stand-in
//! let cfg = r#"{"phoneme_id_map": {"_": [0], "^": [1], "$": [2], "h": [20], "ə": [59]}}"#;
//! let map = text_to_phonemes::phoneme_id_map_from_config(cfg).unwrap();
//! assert_eq!(text_to_phonemes::to_ids("hə", &map), [1, 0, 20, 0, 59, 0, 2]);
//! ```
//!
//! Pipeline: text normalization (numbers, money, times, abbreviations, symbols; sentence and clause marks kept the
//! way Piper keeps espeak's terminators) -> per word the embedded CMUdict lexicon or, out of vocabulary, a
//! letter-context model trained only on CMUdict -> ARPAbet to espeak-style IPA, with reduced function words inside
//! sentences.
//!
//! Licensing: MIT OR Apache-2.0; the embedded CMUdict data is BSD-style (see NOTICE). espeak-ng is not a
//! dependency; it was only used offline, in test scripts, to tune the rules.

use std::collections::HashMap;

mod ipa;
mod lexicon;
mod normalize;
mod numbers;
mod oov;
mod phoneset;

/// Training and evaluation pieces for the bundled `g2p-tool` binary. Not part of the stable API.
#[doc(hidden)]
pub mod __internal {
    pub use crate::oov::{Model, evaluate, is_heldout};
    pub use crate::phoneset::parse_cmudict;
}

pub use normalize::Tok;

static LEXICON_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lexicon.bin"));
static OOV_MODEL: &str = include_str!(concat!(env!("OUT_DIR"), "/oov.model"));

/// Function words read without stress inside a sentence, with their reduced espeak-style forms. Words not listed
/// here keep their dictionary stress.
const FUNCTION_WORDS: &[(&str, &str)] = &[
    ("a", "ɐ"),
    ("an", "ɐn"),
    ("the", "ðə"),
    ("and", "ænd"),
    ("or", "ɔːɹ"),
    ("of", "ʌv"),
    ("to", "tə"),
    ("in", "ɪn"),
    ("on", "ˌɔn"),
    ("at", "æt"),
    ("for", "fɔːɹ"),
    ("with", "wɪð"),
    ("from", "fɹʌm"),
    ("by", "baɪ"),
    ("as", "æz"),
    ("is", "ɪz"),
    ("was", "wʌz"),
    ("are", "ɑːɹ"),
    ("were", "wɜː"),
    ("be", "biː"),
    ("been", "bɪn"),
    ("am", "æm"),
    ("i", "aɪ"),
    ("you", "juː"),
    ("he", "hiː"),
    ("she", "ʃiː"),
    ("it", "ɪt"),
    ("we", "wiː"),
    ("they", "ðeɪ"),
    ("me", "mˌiː"),
    ("him", "hɪm"),
    ("her", "hɜː"),
    ("us", "ʌs"),
    ("them", "ðˌɛm"),
    ("my", "maɪ"),
    ("your", "jʊɹ"),
    ("his", "hɪz"),
    ("its", "ɪts"),
    ("our", "ˌaʊɚ"),
    ("their", "ðɛɹ"),
    ("that", "ðæt"),
    ("does", "dʌz"),
    ("have", "hæv"),
    ("has", "hɐz"),
    ("had", "hæd"),
    ("will", "wɪl"),
    ("would", "wʊd"),
    ("can", "kæn"),
    ("could", "kʊd"),
    ("should", "ʃˌʊd"),
    ("if", "ɪf"),
    ("than", "ðɐn"),
    ("this", "ðɪs"),
    ("when", "wɛn"),
    ("through", "θɹuː"),
    ("into", "ˌɪntʊ"),
    ("i'm", "aɪm"),
    ("i'll", "aɪl"),
    ("i've", "aɪv"),
    ("i'd", "aɪd"),
];

/// Words espeak says alone without stress (the rest of FUNCTION_WORDS keep their stress when said alone).
const ISOLATED_UNSTRESSED: &[(&str, &str)] = &[
    ("a", "ˈeɪ"),
    ("the", "ðə"),
    ("and", "ænd"),
    ("of", "ʌv"),
    ("for", "fɔːɹ"),
    ("to", "tuː"),
    ("an", "æn"),
    ("her", "hɜː"),
    ("were", "wɜː"),
    ("your", "jʊɹ"),
    ("was", "wʌz"),
    ("than", "ðɐn"),
    ("into", "ˌɪntʊ"),
];

/// Words that carry only secondary stress: said alone, and (SENTENCE_SECONDARY) inside sentences.
const ISOLATED_SECONDARY: &[&str] = &["us", "but", "our", "one", "what", "any", "when", "been", "some", "had", "them"];
const SENTENCE_SECONDARY: &[&str] = &[
    "about", "across", "before", "being", "but", "go", "make", "near", "one", "so", "some", "under", "what", "where",
    "which",
];

/// The G2P engine. Cheap to share: the lexicon is used in place; the OOV model is parsed once in `new`.
pub struct G2p {
    lex: lexicon::Lexicon,
    oov: oov::Model,
    function: HashMap<&'static str, &'static str>,
}

impl Default for G2p {
    fn default() -> Self {
        Self::new()
    }
}

impl G2p {
    pub fn new() -> G2p {
        G2p {
            lex: lexicon::Lexicon::from_blob(LEXICON_BLOB),
            oov: oov::Model::from_text(OOV_MODEL),
            function: FUNCTION_WORDS.iter().copied().collect(),
        }
    }

    /// Entries in the embedded lexicon.
    pub fn lexicon_len(&self) -> usize {
        self.lex.len()
    }

    /// Windows in the embedded OOV model.
    pub fn oov_len(&self) -> usize {
        self.oov.len()
    }

    /// Whether a word (lowercase) is in the embedded CMUdict lexicon.
    pub fn known(&self, word: &str) -> bool {
        self.lex.get(word).is_some()
    }

    /// ARPAbet symbols (vowels with stress digits) for a word: the lexicon, then possessive/contraction splits,
    /// then the OOV model. `g2p.arpabet("water")` is `["W", "AO1", "T", "ER0"]`.
    pub fn arpabet(&self, word: &str) -> Vec<String> {
        self.phones(&word.to_lowercase()).into_iter().map(phoneset::phone_sym).collect()
    }

    /// ARPAbet symbols from the out-of-vocabulary model alone (what a word not in CMUdict would get).
    pub fn oov_arpabet(&self, word: &str) -> Vec<String> {
        self.oov.predict(&word.to_lowercase()).into_iter().map(phoneset::phone_sym).collect()
    }

    /// ARPAbet codes for a lowercase word.
    pub(crate) fn phones(&self, word: &str) -> Vec<u8> {
        if let Some(p) = self.lex.get(word) {
            return p.to_vec();
        }
        // possessive 's on a known word
        if let Some(stem) = word.strip_suffix("'s") {
            if let Some(p) = self.lex.get(stem) {
                let mut v = p.to_vec();
                let last = phoneset::phone_sym(*v.last().unwrap());
                let suffix: &[&str] = match last.as_str() {
                    "S" | "Z" | "SH" | "ZH" | "CH" | "JH" => &["IH0", "Z"],
                    "P" | "T" | "K" | "F" | "TH" => &["S"],
                    _ => &["Z"],
                };
                v.extend(suffix.iter().filter_map(|s| phoneset::phone_code(s)));
                return v;
            }
        }
        let bare: String = word.chars().filter(|c| *c != '\'').collect();
        if bare != word {
            if let Some(p) = self.lex.get(&bare) {
                return p.to_vec();
            }
        }
        self.oov.predict(word)
    }

    /// IPA for one word said alone, e.g. "water" -> "wˈɔːɾɚ" (function words unstressed, as espeak says them).
    pub fn word(&self, word: &str) -> String {
        let w = word.to_lowercase();
        if let Some((_, f)) = ISOLATED_UNSTRESSED.iter().find(|(k, _)| *k == w) {
            return f.to_string();
        }
        let ipa = self.plain_ipa(&w);
        if ISOLATED_SECONDARY.contains(&w.as_str()) { ipa.replace('ˈ', "ˌ") } else { ipa }
    }

    fn plain_ipa(&self, w: &str) -> String {
        ipa::to_ipa(&self.phones(w), w, ipa::WordOpts::default())
    }

    fn letter(&self, c: char) -> String {
        let codes: Vec<u8> = if c == 'a' {
            vec![phoneset::phone_code("EY1").unwrap()]
        } else {
            self.phones(&c.to_string())
        };
        ipa::to_ipa(&codes, &c.to_string(), ipa::WordOpts::default())
    }

    /// IPA for a word inside a sentence (function words reduced, some common words with secondary stress only).
    fn word_in_sentence(&self, w: &str) -> String {
        if let Some(f) = self.function.get(w) {
            return f.to_string();
        }
        let ipa = self.plain_ipa(w);
        if SENTENCE_SECONDARY.contains(&w) { ipa.replace('ˈ', "ˌ") } else { ipa }
    }

    /// One sentence's tokens to IPA, the way Piper writes espeak's output: words separated by spaces, clause marks
    /// right after the word they follow. Adds espeak's joins between words: "in the" -> ɪnðə, "the" before a vowel
    /// -> ðɪʲ, a linking ɹ after ɚ and a ʲ after i before a vowel, the flap in "it is", and spelled letters run
    /// together with secondary stress on all but the last ("pm" pˌiːʲˈɛm).
    pub fn sentence_ipa(&self, toks: &[Tok]) -> String {
        struct Part {
            text: String,
            word: Option<String>,
            punct: bool,
        }
        let mut parts: Vec<Part> = Vec::with_capacity(toks.len());
        let mut i = 0;
        while i < toks.len() {
            match &toks[i] {
                Tok::Word(w) => parts.push(Part { text: self.word_in_sentence(w), word: Some(w.clone()), punct: false }),
                Tok::Letter(_) => {
                    let mut letters = Vec::new();
                    while let Some(Tok::Letter(c)) = toks.get(i) {
                        letters.push(self.letter(*c));
                        i += 1;
                    }
                    let n = letters.len();
                    let mut text = String::new();
                    for (k, l) in letters.into_iter().enumerate() {
                        let l = if k + 1 < n { l.replace('ˈ', "ˌ") } else { l };
                        if text.ends_with('ː') && text.ends_with("iː") && starts_with_vowel(&l) {
                            text.push('ʲ');
                        }
                        text.push_str(&l);
                    }
                    parts.push(Part { text, word: None, punct: false });
                    continue;
                }
                Tok::Punct(c) => parts.push(Part { text: c.to_string(), word: None, punct: true }),
            }
            i += 1;
        }
        // joins that depend on the next word (punctuation breaks them)
        for k in 0..parts.len().saturating_sub(1) {
            if parts[k].punct || parts[k + 1].punct {
                continue;
            }
            let next_vowel = starts_with_vowel(&parts[k + 1].text);
            let w = parts[k].word.clone().unwrap_or_default();
            let next_w = parts[k + 1].word.clone().unwrap_or_default();
            let cur = &mut parts[k].text;
            if w == "the" && next_vowel {
                *cur = "ðɪʲ".into();
            } else if self.function.contains_key(w.as_str()) && next_vowel && cur.ends_with('t') {
                // "of it at" ʌv ɪɾ æt
                let stem = cur[..cur.len() - 1].to_string();
                if stem.chars().last().is_some_and(is_ipa_vowel) {
                    *cur = format!("{stem}ɾ");
                }
            } else if next_vowel && cur.ends_with('ɚ') {
                cur.push('ɹ'); // darker at: dˈɑːɹkɚɹ æt
            } else if next_vowel && cur.ends_with('i') {
                cur.push('ʲ'); // only a: ˈoʊnliʲ ɐ
            }
            if (w == "in" || w == "on") && next_w == "the" {
                parts[k].text = if w == "in" { "ɪn".into() } else { "ɔn".into() };
                parts[k].punct = false;
                parts[k + 1].word = Some("\u{1}join".into());
            }
        }
        let mut s = String::new();
        for p in parts {
            let join = p.word.as_deref() == Some("\u{1}join");
            if !p.punct && !join && !s.is_empty() {
                s.push(' ');
            }
            s.push_str(&p.text);
        }
        s
    }

    /// Normalized sentences (tokens), for streaming TTS sentence by sentence.
    pub fn normalize(&self, text: &str) -> Vec<Vec<Tok>> {
        normalize::normalize(text, &|w| self.known(w))
    }

    /// IPA per sentence.
    pub fn sentences(&self, text: &str) -> Vec<String> {
        self.normalize(text).iter().map(|s| self.sentence_ipa(s)).collect()
    }
}

/// A Piper voice's `phoneme_id_map` (keys are code points, or multi-code-point clusters such as "aɪ") and its
/// optional `vowel_clusters` (newer Piper configs: runs of code points merged before the lookup).
#[derive(Clone, Debug, Default)]
pub struct PhonemeIdMap {
    pub ids: HashMap<String, Vec<i64>>,
    /// Clusters to merge, longest first.
    pub clusters: Vec<Vec<char>>,
}

impl PhonemeIdMap {
    pub fn get(&self, phoneme: &str) -> Option<&[i64]> {
        self.ids.get(phoneme).map(|v| v.as_slice())
    }

    pub fn contains(&self, c: char) -> bool {
        let mut b = [0u8; 4];
        self.ids.contains_key(c.encode_utf8(&mut b) as &str)
    }
}

/// Read `phoneme_id_map` and `vowel_clusters` from a Piper voice config (`<voice>.onnx.json`).
pub fn phoneme_id_map_from_config(json: &str) -> Result<PhonemeIdMap, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let m = v.get("phoneme_id_map").and_then(|m| m.as_object()).ok_or("no phoneme_id_map")?;
    let mut out = PhonemeIdMap::default();
    for (k, ids) in m {
        let ids: Vec<i64> = ids.as_array().ok_or("ids not an array")?.iter().filter_map(|x| x.as_i64()).collect();
        out.ids.insert(k.clone(), ids);
    }
    if let Some(vc) = v.get("vowel_clusters").and_then(|x| x.as_array()) {
        for c in vc {
            let cl: Vec<char> = match c {
                serde_json::Value::Array(a) => a.iter().filter_map(|s| s.as_str()).flat_map(|s| s.chars()).collect(),
                serde_json::Value::String(s) => s.chars().collect(),
                _ => continue,
            };
            if cl.len() > 1 {
                out.clusters.push(cl);
            }
        }
        out.clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    }
    Ok(out)
}

pub const BOS: &str = "^";
pub const EOS: &str = "$";
pub const PAD: &str = "_";

/// Split IPA into Piper's phonemes: NFD code points, then known vowel clusters merged (greedy, longest first).
pub fn piper_phonemes(phonemes: &str, map: &PhonemeIdMap) -> Vec<String> {
    use unicode_normalization::UnicodeNormalization;
    let chars: Vec<char> = phonemes.nfd().collect();
    let mut out = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        if let Some(cl) = map.clusters.iter().find(|cl| chars[i..].starts_with(cl)) {
            out.push(cl.iter().collect());
            i += cl.len();
        } else {
            out.push(chars[i].to_string());
            i += 1;
        }
    }
    out
}

/// Piper's phoneme ids for one sentence (piper's phonemes_to_ids): BOS, PAD, then each phoneme's ids followed by
/// PAD, then EOS. Phonemes missing from the map are skipped, as Piper does.
pub fn to_ids(phonemes: &str, map: &PhonemeIdMap) -> Vec<i64> {
    let get = |p: &str| map.get(p).unwrap_or(&[]);
    let mut ids: Vec<i64> = Vec::with_capacity(2 * phonemes.len() + 3);
    ids.extend_from_slice(get(BOS));
    ids.extend_from_slice(get(PAD));
    for p in piper_phonemes(phonemes, map) {
        if let Some(v) = map.get(&p) {
            ids.extend_from_slice(v);
            ids.extend_from_slice(get(PAD));
        }
    }
    ids.extend_from_slice(get(EOS));
    ids
}

/// Phonemes of `phonemes` that the map lacks.
pub fn missing_phonemes(phonemes: &str, map: &PhonemeIdMap) -> Vec<String> {
    let mut v: Vec<String> = piper_phonemes(phonemes, map).into_iter().filter(|p| map.get(p).is_none()).collect();
    v.sort();
    v.dedup();
    v
}

fn is_ipa_vowel(c: char) -> bool {
    "aeiouæɐɑɒɔəɚɛɜɝɪʊʌᵻː".contains(c)
}

fn starts_with_vowel(ipa: &str) -> bool {
    ipa.chars().find(|c| *c != 'ˈ' && *c != 'ˌ').is_some_and(is_ipa_vowel)
}
