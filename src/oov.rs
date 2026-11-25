//! Out-of-vocabulary fallback: a letter-context model trained only on CMUdict.
//!
//! Training: (1) align each word's letters to its phones (each letter emits 0, 1 or 2 phones) with hard-EM
//! (Viterbi) re-estimation of p(phones | letter); (2) for every letter, count the aligned output (phones with
//! stress) in a nested chain of letter windows, widest first: (4 left, 4 right), (3,4), (3,3), (2,3), (2,2), (1,2),
//! (1,1), (0,1), (0,0); (3) keep a window only if it was seen `min_count` times and its best output differs from
//! what the next narrower window would say. Prediction walks the chain and takes the first window it knows, then
//! repairs stress (exactly one primary).

use std::collections::HashMap;

use crate::phoneset::{CONSONANTS, phone_code, phone_sym};

/// (left, right) letter windows, widest first; each is contained in the one before it.
pub const LEVELS: [(usize, usize); 9] = [(4, 4), (3, 4), (3, 3), (2, 3), (2, 2), (1, 2), (1, 1), (0, 1), (0, 0)];
const PAD: u8 = 28;
const NLET: usize = 29;

fn letter_id(c: u8) -> Option<u8> {
    match c {
        b'a'..=b'z' => Some(c - b'a' + 1),
        b'\'' => Some(27),
        _ => None,
    }
}

fn letter_char(id: u8) -> char {
    match id {
        1..=26 => (b'a' + id - 1) as char,
        27 => '\'',
        _ => '#',
    }
}

/// A label: 0 = no phone, c1*70 + c2 (c2 = 0 for one phone).
type Label = u16;

fn label_phones(l: Label) -> Vec<u8> {
    let (a, b) = ((l / 70) as u8, (l % 70) as u8);
    [a, b].into_iter().filter(|&c| c != 0).collect()
}

fn label_of(ph: &[u8]) -> Label {
    match ph {
        [] => 0,
        [a] => *a as u16 * 70,
        [a, b] => *a as u16 * 70 + *b as u16,
        _ => unreachable!(),
    }
}

fn key(level: usize, padded: &[u8], i: usize) -> u64 {
    let (l, r) = LEVELS[level];
    let mut k: u64 = level as u64;
    for &c in &padded[i - l..=i + r] {
        k = (k << 5) | c as u64;
    }
    k | ((level as u64) << 60)
}

fn pad_word(word: &str) -> Option<Vec<u8>> {
    let mut v = vec![PAD; 4];
    for c in word.bytes() {
        v.push(letter_id(c.to_ascii_lowercase())?);
    }
    if v.len() == 4 {
        return None;
    }
    v.extend_from_slice(&[PAD; 4]);
    Some(v)
}

fn is_vowel(code: u8) -> bool {
    code as usize > CONSONANTS.len()
}

fn base(code: u8) -> usize {
    let i = code as usize - 1;
    if i < CONSONANTS.len() { i } else { CONSONANTS.len() + (i - CONSONANTS.len()) / 3 }
}

fn stress(code: u8) -> u8 {
    ((code as usize - 1 - CONSONANTS.len()) % 3) as u8
}

fn with_stress(code: u8, s: u8) -> u8 {
    let v = (code as usize - 1 - CONSONANTS.len()) / 3;
    (CONSONANTS.len() + v * 3 + s as usize) as u8 + 1
}

const NB: usize = 39; // base phones
const NCHUNK: usize = 1 + NB + NB * NB;

/// Viterbi alignment of letters to phones: per letter, how many phones it emits (0..=2).
fn align(letters: &[u8], bases: &[usize], logp: &[f32]) -> Option<Vec<u8>> {
    let (n, m) = (letters.len(), bases.len());
    if m > 2 * n {
        return None;
    }
    let w = m + 1;
    let mut dp = vec![f32::NEG_INFINITY; (n + 1) * w];
    let mut back = vec![0u8; (n + 1) * w];
    dp[0] = 0.0;
    for i in 1..=n {
        let row = letters[i - 1] as usize * NCHUNK;
        for j in 0..=m {
            let mut best = f32::NEG_INFINITY;
            let mut bk = 0u8;
            for k in 0..=2usize.min(j) {
                let prev = dp[(i - 1) * w + j - k];
                if prev == f32::NEG_INFINITY {
                    continue;
                }
                let chunk = match k {
                    0 => 0,
                    1 => 1 + bases[j - 1],
                    _ => 1 + NB + bases[j - 2] * NB + bases[j - 1],
                };
                let s = prev + logp[row + chunk];
                if s > best {
                    best = s;
                    bk = k as u8;
                }
            }
            dp[i * w + j] = best;
            back[i * w + j] = bk;
        }
    }
    if dp[n * w + m] == f32::NEG_INFINITY {
        return None;
    }
    let mut out = vec![0u8; n];
    let mut j = m;
    for i in (1..=n).rev() {
        let k = back[i * w + j];
        out[i - 1] = k;
        j -= k as usize;
    }
    Some(out)
}

/// One aligned word: padded letter ids and, per letter, its output label.
struct Aligned {
    padded: Vec<u8>,
    labels: Vec<Label>,
}

fn align_all(words: &[(String, Vec<u8>)]) -> Vec<Aligned> {
    let prepared: Vec<(Vec<u8>, Vec<usize>, &Vec<u8>)> = words
        .iter()
        .filter_map(|(w, ph)| {
            let p = pad_word(w)?;
            let letters = p[4..p.len() - 4].to_vec();
            Some((letters, ph.iter().map(|&c| base(c)).collect(), ph))
        })
        .collect();
    // init: letter/phone co-occurrence near the diagonal
    let mut cooc = vec![0.1f32; NLET * NB];
    for (letters, bases, _) in &prepared {
        let (n, m) = (letters.len() as f32, bases.len() as f32);
        for (i, &l) in letters.iter().enumerate() {
            for (j, &b) in bases.iter().enumerate() {
                let d = (i as f32 + 0.5) / n - (j as f32 + 0.5) / m;
                if d.abs() < 0.25 {
                    cooc[l as usize * NB + b] += 1.0;
                }
            }
        }
    }
    let mut logp = vec![f32::NEG_INFINITY; NLET * NCHUNK];
    for l in 0..NLET {
        let tot: f32 = cooc[l * NB..(l + 1) * NB].iter().sum();
        let lp: Vec<f32> = (0..NB).map(|b| (cooc[l * NB + b] / tot).ln()).collect();
        logp[l * NCHUNK] = (0.15f32).ln();
        for b in 0..NB {
            logp[l * NCHUNK + 1 + b] = lp[b];
            for b2 in 0..NB {
                logp[l * NCHUNK + 1 + NB + b * NB + b2] = lp[b] + lp[b2] - 3.0;
            }
        }
    }
    let mut result = Vec::new();
    for iter in 0..6 {
        let mut counts = vec![0.0f32; NLET * NCHUNK];
        let last = iter == 5;
        for (letters, bases, ph) in &prepared {
            let Some(a) = align(letters, bases, &logp) else { continue };
            let mut j = 0;
            let mut labels = Vec::with_capacity(letters.len());
            for (i, &k) in a.iter().enumerate() {
                let chunk = match k {
                    0 => 0,
                    1 => 1 + bases[j],
                    _ => 1 + NB + bases[j] * NB + bases[j + 1],
                };
                counts[letters[i] as usize * NCHUNK + chunk] += 1.0;
                if last {
                    labels.push(label_of(&ph[j..j + k as usize]));
                }
                j += k as usize;
            }
            if last {
                let mut padded = vec![PAD; 4];
                padded.extend_from_slice(letters);
                padded.extend_from_slice(&[PAD; 4]);
                result.push(Aligned { padded, labels });
            }
        }
        if !last {
            for l in 0..NLET {
                let row = &counts[l * NCHUNK..(l + 1) * NCHUNK];
                let tot: f32 = row.iter().sum::<f32>() + 0.01 * NCHUNK as f32;
                for c in 0..NCHUNK {
                    // pairs are rare: a stronger floor keeps unseen pairs unlikely
                    let floor = if c > NB { 0.0005 } else { 0.01 };
                    logp[l * NCHUNK + c] = ((row[c] + floor) / tot).ln();
                }
            }
        }
    }
    result
}

/// The trained model: window key -> output label.
pub struct Model {
    table: HashMap<u64, Label>,
}

impl Model {
    pub fn empty() -> Model {
        Model { table: HashMap::new() }
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Train on (word, phone codes); keep windows seen at least `min_count` times (the two narrowest always).
    pub fn train(words: &[(String, Vec<u8>)], min_count: u32) -> Model {
        let aligned = align_all(words);
        let mut counts: HashMap<(u64, Label), u32> = HashMap::new();
        for a in &aligned {
            for (p, &lab) in a.labels.iter().enumerate() {
                let i = p + 4;
                for lv in 0..LEVELS.len() {
                    *counts.entry((key(lv, &a.padded, i), lab)).or_insert(0) += 1;
                }
            }
        }
        // best label and total per window
        let mut best: HashMap<u64, (Label, u32, u32)> = HashMap::new();
        for (&(k, lab), &c) in &counts {
            let e = best.entry(k).or_insert((lab, 0, 0));
            e.2 += c;
            if c > e.1 || (c == e.1 && lab < e.0) {
                e.0 = lab;
                e.1 = c;
            }
        }
        drop(counts);
        // prune narrowest first; eff = what prediction says for a window after pruning
        let mut eff: HashMap<u64, Label> = HashMap::new();
        let mut table: HashMap<u64, Label> = HashMap::new();
        for a in &aligned {
            for p in 0..a.labels.len() {
                let i = p + 4;
                let mut parent: Option<Label> = None;
                for lv in (0..LEVELS.len()).rev() {
                    let k = key(lv, &a.padded, i);
                    let e = if let Some(&e) = eff.get(&k) {
                        e
                    } else {
                        let (lab, _, tot) = best[&k];
                        let keep = match parent {
                            None => true,
                            Some(pl) => lab != pl && (tot >= min_count || lv >= LEVELS.len() - 2),
                        };
                        let e = if keep {
                            table.insert(k, lab);
                            lab
                        } else {
                            parent.unwrap()
                        };
                        eff.insert(k, e);
                        e
                    };
                    parent = Some(e);
                }
            }
        }
        Model { table }
    }

    /// Phone codes for a word (letters a-z and '; anything else is skipped).
    pub fn predict(&self, word: &str) -> Vec<u8> {
        let clean: String = word.chars().filter(|c| c.is_ascii_alphabetic() || *c == '\'').collect();
        let Some(padded) = pad_word(&clean) else { return Vec::new() };
        let mut out = Vec::new();
        for i in 4..padded.len() - 4 {
            for lv in 0..LEVELS.len() {
                if let Some(&lab) = self.table.get(&key(lv, &padded, i)) {
                    out.extend(label_phones(lab));
                    break;
                }
            }
        }
        fix_stress(&mut out);
        out
    }

    /// Text form: `level<TAB>window letters<TAB>phones` (phones "-" for none, "K+S" for two).
    pub fn to_text(&self) -> String {
        let mut lines: Vec<String> = self
            .table
            .iter()
            .map(|(&k, &lab)| {
                let level = (k >> 60) as usize;
                let (l, r) = LEVELS[level];
                let n = l + 1 + r;
                let chars: String =
                    (0..n).rev().map(|s| letter_char(((k >> (5 * s)) & 31) as u8)).collect();
                let ph = label_phones(lab);
                let phs = if ph.is_empty() {
                    "-".to_string()
                } else {
                    ph.iter().map(|&c| phone_sym(c)).collect::<Vec<_>>().join("+")
                };
                format!("{level}\t{chars}\t{phs}")
            })
            .collect();
        lines.sort();
        let mut s = String::from("# text-to-phonemes OOV letter-context model, trained only on CMUdict (LICENSE-CMUDICT)\n");
        for l in lines {
            s.push_str(&l);
            s.push('\n');
        }
        s
    }

    pub fn from_text(text: &str) -> Model {
        let mut table = HashMap::new();
        for line in text.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let mut it = line.split('\t');
            let (Some(lv), Some(chars), Some(phs)) = (it.next(), it.next(), it.next()) else { continue };
            let Ok(level) = lv.parse::<usize>() else { continue };
            let mut k: u64 = level as u64;
            for c in chars.bytes() {
                let id = if c == b'#' { PAD } else { letter_id(c).unwrap_or(PAD) };
                k = (k << 5) | id as u64;
            }
            k |= (level as u64) << 60;
            let codes: Vec<u8> =
                if phs == "-" { Vec::new() } else { phs.split('+').filter_map(phone_code).collect() };
            if codes.len() <= 2 {
                table.insert(k, label_of(&codes));
            }
        }
        Model { table }
    }
}

/// Exactly one primary stress: demote extra primaries to secondary; with none, promote the first secondary, else the
/// first vowel.
pub fn fix_stress(ph: &mut [u8]) {
    let mut seen = false;
    for c in ph.iter_mut() {
        if is_vowel(*c) && stress(*c) == 1 {
            if seen {
                *c = with_stress(*c, 2);
            }
            seen = true;
        }
    }
    if !seen {
        let pick = ph
            .iter()
            .position(|&c| is_vowel(c) && stress(c) == 2)
            .or_else(|| ph.iter().position(|&c| is_vowel(c)));
        if let Some(i) = pick {
            ph[i] = with_stress(ph[i], 1);
        }
    }
}

/// Levenshtein distance between phone sequences.
pub fn edit_distance(a: &[u8], b: &[u8]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (x != y) as usize).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Held-out split: about 1 word in 20, by a fixed hash of the word.
pub fn is_heldout(word: &str) -> bool {
    let mut h: u32 = 2166136261;
    for b in word.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    h % 20 == 0
}

/// PER (with stress, and stress-free) and word accuracy of `model` on `test`.
pub fn evaluate(model: &Model, test: &[(String, Vec<u8>)]) -> (f64, f64, f64) {
    let (mut err, mut err_ns, mut total, mut right) = (0usize, 0usize, 0usize, 0usize);
    let strip = |v: &[u8]| -> Vec<u8> { v.iter().map(|&c| base(c) as u8).collect() };
    for (w, ref_ph) in test {
        let hyp = model.predict(w);
        let d = edit_distance(&hyp, ref_ph);
        err += d;
        err_ns += edit_distance(&strip(&hyp), &strip(ref_ph));
        total += ref_ph.len();
        right += (d == 0) as usize;
    }
    let t = total.max(1) as f64;
    (err as f64 / t, err_ns as f64 / t, right as f64 / test.len().max(1) as f64)
}
