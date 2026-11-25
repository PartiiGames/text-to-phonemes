//! The embedded CMUdict table: binary search over front-coded blocks, no parsing at load.

use crate::phoneset::{LEX_BLOCK, LEX_MAGIC};

pub struct Lexicon {
    blob: &'static [u8],
    entries: usize,
    blocks: usize,
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

impl Lexicon {
    pub fn from_blob(blob: &'static [u8]) -> Lexicon {
        if blob.len() < 12 || &blob[..4] != LEX_MAGIC {
            return Lexicon { blob: &[], entries: 0, blocks: 0 };
        }
        Lexicon { blob, entries: u32_at(blob, 4), blocks: u32_at(blob, 8) }
    }

    pub fn len(&self) -> usize {
        self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    fn data_start(&self) -> usize {
        12 + 4 * self.blocks
    }

    /// The whole first word of block `b` (it has no shared prefix).
    fn block_head(&self, b: usize) -> &'static [u8] {
        let p = self.data_start() + u32_at(self.blob, 12 + 4 * b);
        let n = self.blob[p + 1] as usize;
        &self.blob[p + 2..p + 2 + n]
    }

    /// Phone codes for a lowercase word.
    pub fn get(&self, word: &str) -> Option<&'static [u8]> {
        if self.blocks == 0 {
            return None;
        }
        let w = word.as_bytes();
        // last block whose head <= w
        let (mut lo, mut hi) = (0usize, self.blocks);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.block_head(mid) <= w {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mut p = self.data_start() + u32_at(self.blob, 12 + 4 * lo);
        let mut cur: Vec<u8> = Vec::with_capacity(32);
        let n = LEX_BLOCK.min(self.entries - lo * LEX_BLOCK);
        for _ in 0..n {
            let shared = self.blob[p] as usize;
            let slen = self.blob[p + 1] as usize;
            cur.truncate(shared);
            cur.extend_from_slice(&self.blob[p + 2..p + 2 + slen]);
            p += 2 + slen;
            let nph = self.blob[p] as usize;
            let ph = &self.blob[p + 1..p + 1 + nph];
            p += 1 + nph;
            match cur.as_slice().cmp(w) {
                std::cmp::Ordering::Equal => return Some(ph),
                std::cmp::Ordering::Greater => return None,
                std::cmp::Ordering::Less => {}
            }
        }
        None
    }

    /// Every (word, codes), in order (for training and tests).
    pub fn iter(&self) -> impl Iterator<Item = (String, &'static [u8])> + '_ {
        let mut p = self.data_start();
        let mut cur: Vec<u8> = Vec::new();
        (0..self.entries).map(move |_| {
            let shared = self.blob[p] as usize;
            let slen = self.blob[p + 1] as usize;
            cur.truncate(shared);
            cur.extend_from_slice(&self.blob[p + 2..p + 2 + slen]);
            p += 2 + slen;
            let nph = self.blob[p] as usize;
            let ph = &self.blob[p + 1..p + 1 + nph];
            p += 1 + nph;
            (String::from_utf8_lossy(&cur).into_owned(), ph)
        })
    }
}
