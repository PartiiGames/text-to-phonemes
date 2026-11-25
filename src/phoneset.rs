// ARPAbet phone codes and the compact lexicon format. Shared by build.rs (include!) and the library, so this file
// uses only std and defines no `use` of crate items.
//
// A phone code is one byte, 1..=69: the 24 consonants, then the 15 vowels x stress 0/1/2.
// Lexicon blob ("G2PL"): magic, u32 entries, u32 blocks, then a u32 offset per block of BLOCK entries, then the
// entries. Entries are sorted by word bytes and front-coded inside a block (the block's first entry is whole):
//   [shared prefix len u8][suffix len u8][suffix bytes][phone count u8][phone codes]

pub const CONSONANTS: [&str; 24] = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W",
    "Y", "Z", "ZH",
];
pub const VOWELS: [&str; 15] =
    ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"];
pub const LEX_BLOCK: usize = 16;
pub const LEX_MAGIC: &[u8; 4] = b"G2PL";

/// ARPAbet symbol (vowels with a stress digit, e.g. "AH0") to its code; None if unknown.
pub fn phone_code(sym: &str) -> Option<u8> {
    if let Some(i) = CONSONANTS.iter().position(|c| *c == sym) {
        return Some(i as u8 + 1);
    }
    let (base, stress) = match sym.as_bytes().last() {
        Some(d @ b'0'..=b'2') => (&sym[..sym.len() - 1], (d - b'0') as usize),
        _ => (sym, 1), // a vowel without a digit: primary stress
    };
    let v = VOWELS.iter().position(|c| *c == base)?;
    Some((CONSONANTS.len() + v * 3 + stress) as u8 + 1)
}

/// Code back to its ARPAbet symbol (vowels with stress digit).
pub fn phone_sym(code: u8) -> String {
    let i = code as usize - 1;
    if i < CONSONANTS.len() {
        CONSONANTS[i].to_string()
    } else {
        let v = i - CONSONANTS.len();
        format!("{}{}", VOWELS[v / 3], v % 3)
    }
}

/// Parse a CMUdict file (cmudict.dict format: `word PH ON ES`, variants `word(2)`, `# comments`) into sorted
/// (word, codes), first pronunciation only.
pub fn parse_cmudict(text: &str) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(word) = it.next() else { continue };
        if word.ends_with(')') {
            continue; // alternate pronunciation
        }
        let codes: Option<Vec<u8>> = it.map(phone_code).collect();
        let Some(codes) = codes else { continue };
        if codes.is_empty() || codes.len() > 255 || word.len() > 255 {
            continue;
        }
        out.push((word.to_lowercase(), codes));
    }
    out.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    out.dedup_by(|a, b| a.0 == b.0);
    out
}

/// Encode sorted entries as the lexicon blob.
pub fn encode_lexicon(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    let nblocks = entries.len().div_ceil(LEX_BLOCK);
    let mut data: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::with_capacity(nblocks);
    for (i, (w, ph)) in entries.iter().enumerate() {
        let w = w.as_bytes();
        let shared = if i % LEX_BLOCK == 0 {
            offsets.push(data.len() as u32);
            0
        } else {
            let p = entries[i - 1].0.as_bytes();
            p.iter().zip(w).take_while(|(a, b)| a == b).count().min(255)
        };
        data.push(shared as u8);
        data.push((w.len() - shared) as u8);
        data.extend_from_slice(&w[shared..]);
        data.push(ph.len() as u8);
        data.extend_from_slice(ph);
    }
    let mut blob = Vec::with_capacity(12 + 4 * nblocks + data.len());
    blob.extend_from_slice(LEX_MAGIC);
    blob.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(nblocks as u32).to_le_bytes());
    for o in offsets {
        blob.extend_from_slice(&o.to_le_bytes());
    }
    blob.extend_from_slice(&data);
    blob
}
