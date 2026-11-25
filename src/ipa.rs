//! ARPAbet (CMUdict, with stress) to espeak-ng en-us style IPA.
//!
//! espeak-ng's en-us output conventions this follows (tuned against Piper's phonemizer on the farm,
//! tools/compare.py): the stress mark goes right before the stressed vowel (həlˈoʊ), long vowels carry ː (ɑː ɔː iː
//! uː ɜː), unstressed ER is ɚ and stressed ER is ɜː (ɜːɹ before a vowel), word-initial unstressed "a" is ɐ, the
//! unstressed -ed/-es ending vowel is ᵻ, t/d between a vowel and an unstressed vowel is the flap ɾ, and "-tten/-tton"
//! is ʔn̩.

use crate::phoneset::CONSONANTS;

#[derive(Clone, Copy, PartialEq, Debug)]
struct P {
    base: &'static str,
    stress: u8, // 0,1,2 for vowels, 9 for consonants
}

impl P {
    fn vowel(&self) -> bool {
        self.stress != 9
    }
}

fn decode(code: u8) -> P {
    let i = code as usize - 1;
    if i < CONSONANTS.len() {
        P { base: CONSONANTS[i], stress: 9 }
    } else {
        let v = i - CONSONANTS.len();
        P { base: crate::phoneset::VOWELS[v / 3], stress: (v % 3) as u8 }
    }
}

fn consonant(b: &str) -> &'static str {
    match b {
        "B" => "b",
        "CH" => "tʃ",
        "D" => "d",
        "DH" => "ð",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "JH" => "dʒ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "P" => "p",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        _ => "",
    }
}

/// Options for one word.
#[derive(Clone, Copy, Debug, Default)]
pub struct WordOpts {
    /// Drop stress marks (function words inside a sentence).
    pub unstressed: bool,
}

/// IPA for one word's phone codes; `spelling` is the lowercase written word (some choices depend on it).
pub fn to_ipa(codes: &[u8], spelling: &str, opts: WordOpts) -> String {
    let ph: Vec<P> = codes.iter().map(|&c| decode(c)).collect();
    let n = ph.len();
    let sp = spelling.as_bytes();
    let vowels: Vec<usize> = (0..n).filter(|&i| ph[i].vowel()).collect();
    // Effective stress, the way espeak places it: one vowel is always primary; with several primaries (pm, compounds)
    // the last one stays primary; secondary stress after the primary is dropped (contact kˈɑːntækt), except on aɪ
    // (copyright kˈɑːpɪɹˌaɪt).
    let mut st: Vec<u8> = ph.iter().map(|p| p.stress).collect();
    if vowels.len() == 1 {
        st[vowels[0]] = 1;
    }
    let primaries: Vec<usize> = vowels.iter().copied().filter(|&i| st[i] == 1).collect();
    if let Some((&last, rest)) = primaries.split_last() {
        for &i in rest {
            st[i] = 2;
        }
        for &i in &vowels {
            if i > last && st[i] == 2 && ph[i].base != "AY" {
                st[i] = 0;
            }
        }
    } else if let Some(&i) = vowels.iter().find(|&&i| st[i] == 2).or(vowels.first()) {
        st[i] = 1;
    }
    let last_vowel = vowels.last().copied();
    let is_v = |i: usize| i < n && ph[i].vowel();
    let ends = |sufs: &[&str]| sufs.iter().any(|s| spelling.ends_with(s));
    // the unstressed vowel of an -ed / -es ending: wanted wˈɔntᵻd, roses ɹˈoʊzᵻz
    let suffix_ed_es = |i: usize| {
        i + 2 == n && (ph[i + 1].base == "D" || ph[i + 1].base == "Z") && ends(&["ed", "es"])
    };
    // re- / de- / be- right before the stress: remember ɹᵻmˈɛmbɚ, decide dᵻsˈaɪd, research ɹᵻsˈɜːtʃ
    let prefix = |i: usize| {
        i == 1
            && ["re", "de", "be"].iter().any(|p| spelling.starts_with(p))
            && spelling != "because"
            && ["R", "D", "B"].contains(&ph[0].base)
            && !is_v(2)
            && vowels.get(1).is_some_and(|&j| st[j] == 1)
    };
    let last_syllable = |i: usize| Some(i) == last_vowel;
    let mut out = String::new();
    for i in 0..n {
        let p = ph[i];
        let s = st[i];
        let next = ph.get(i + 1).copied();
        let prev = if i > 0 { Some(ph[i - 1]) } else { None };
        if p.vowel() {
            if !opts.unstressed {
                match s {
                    1 => out.push('ˈ'),
                    2 => out.push('ˌ'),
                    _ => {}
                }
            }
            let before = |bases: &[&str]| next.is_some_and(|q| bases.contains(&q.base));
            let short_o_context = before(&["S", "F", "TH", "NG"]);
            let v = match (p.base, s) {
                ("AA", _) if short_o_context && spelling.contains('o') => "ɔ",
                ("AA", _) if spelling.starts_with("wan") => "ɔ",
                ("AA", _) => "ɑː",
                ("AE", _) => "æ",
                ("AH", 0) if suffix_ed_es(i) || prefix(i) => "ᵻ",
                ("AH", 0) if last_syllable(i) && ends(&["et", "ets", "ite", "ites"]) => "ᵻ",
                ("AH", 0)
                    if last_syllable(i)
                        && ends(&["ice", "ices", "iced", "age", "ages", "aged", "it", "its", "id", "ic", "ics", "ish"]) =>
                {
                    "ɪ"
                }
                // -ity, -icy, -ily: the vowel before the final i
                ("AH", 0)
                    if ends(&["ity", "icy", "ily", "ities", "ilies"])
                        && i + 3 < n + 1
                        && vowels.iter().rev().nth(1) == Some(&i) =>
                {
                    "ɪ"
                }
                ("AH", 0) => {
                    if i == 0 && sp.first() == Some(&b'a') && n > 1 && !spelling.starts_with("acr") {
                        "ɐ"
                    } else {
                        "ə"
                    }
                }
                ("AH", _) => "ʌ",
                ("AO", _) if short_o_context => "ɔ",
                ("AO", _) => "ɔː",
                ("AW", _) => "aʊ",
                ("AY", _) => "aɪ",
                ("EH", _) => "ɛ",
                // before a vowel the r is heard: general dʒˈɛnɚɹəl
                ("ER", 0) if is_v(i + 1) => "ɚɹ",
                ("ER", 0) => "ɚ",
                ("ER", _) if is_v(i + 1) => "ɜːɹ",
                ("ER", _) => "ɜː",
                ("EY", _) => "eɪ",
                ("IH", 0) if suffix_ed_es(i) || prefix(i) => "ᵻ",
                ("IH", _) => "ɪ",
                // here hˈɪɹ
                ("IY", _) if before(&["R"]) => "ɪ",
                // idea aɪdˈiə, really ɹˈiəli
                ("IY", _) if next.is_some_and(|q| q.base == "AH" && q.stress == 0) => "i",
                ("IY", 0) if prefix(i) => "ᵻ",
                // before a vowel espeak writes a glide: being bˈiːʲɪŋ, react ɹɪʲˈækt
                ("IY", 0) if is_v(i + 1) => "ɪʲ",
                ("IY", 0) => "i",
                ("IY", _) if is_v(i + 1) => "iːʲ",
                ("IY", _) => "iː",
                ("OW", _) => "oʊ",
                ("OY", _) => "ɔɪ",
                ("UH", _) => "ʊ",
                ("UW", _) => "uː",
                _ => "",
            };
            out.push_str(v);
            continue;
        }
        let b = p.base;
        let after_vowel = prev.is_some_and(|q| q.vowel() || q.base == "R");
        let before_unstressed = next.is_some_and(|q| q.vowel() && st[i + 1] == 0);
        // t flaps after a vowel or r (party pˈɑːɹɾi), not after n (winter wˈɪntɚ); d stays (ladder lˈædɚ) unless the
        // spelling has a t there (thirty θˈɜːɾi)
        if b == "T" || (b == "D" && spelling.contains('t') && !spelling.contains('d')) {
            // -tten / -tton: ʔn̩
            if b == "T"
                && after_vowel
                && i + 3 == n
                && next.is_some_and(|q| q.base == "AH" && q.stress == 0)
                && ph[i + 2].base == "N"
            {
                out.push_str("ʔn̩");
                break;
            }
            if after_vowel && before_unstressed && i > 0 {
                out.push('ɾ');
                continue;
            }
        }
        if b == "R" && prev.is_some_and(|q| q.base == "ER") {
            continue; // ER already wrote its r colour
        }
        out.push_str(consonant(b));
    }
    out
}
