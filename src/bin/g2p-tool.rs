//! g2p-tool: train the OOV model and feed the farm comparison (tools/compare.py).
//!   g2p-tool train <cmudict.dict> <out.model> [min_count]   held-out PER (1 word in 20), then train on all and write
//!   g2p-tool words    stdin: one word per line -> "word<TAB>ipa" (the word said alone)
//!   g2p-tool sents    stdin: one text per line -> its sentences' IPA joined by TAB
//!   g2p-tool ids <voice.onnx.json>   stdin: one IPA string per line -> its Piper ids as JSON
//!   g2p-tool stats    embedded lexicon entries and model size
use std::io::{BufRead, Write};
use std::time::Instant;

use text_to_phonemes::G2p;
use text_to_phonemes::__internal as oov;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    match cmd {
        "train" => {
            let text = std::fs::read_to_string(&args[2]).expect("read cmudict");
            let all = oov::parse_cmudict(&text);
            // the model must spell out real words: skip entries with digits, dots or other marks
            let all: Vec<_> =
                all.into_iter().filter(|(w, _)| w.bytes().all(|c| c.is_ascii_lowercase() || c == b'\'')).collect();
            let (test, train): (Vec<_>, Vec<_>) = all.iter().cloned().partition(|(w, _)| oov::is_heldout(w));
            println!("entries {} train {} heldout {}", all.len(), train.len(), test.len());
            let chosen: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
            for mc in [1u32, 2, 3, 5] {
                let t0 = Instant::now();
                let m = oov::Model::train(&train, mc);
                let (per, per_ns, wacc) = oov::evaluate(&m, &test);
                println!(
                    "min_count {mc}: windows {} text {} bytes  heldout PER {:.2}% (no stress {:.2}%) word acc {:.2}%  [{:.1}s]",
                    m.len(),
                    m.to_text().len(),
                    per * 100.0,
                    per_ns * 100.0,
                    wacc * 100.0,
                    t0.elapsed().as_secs_f64()
                );
            }
            let m = oov::Model::train(&all, chosen);
            let txt = m.to_text();
            std::fs::write(&args[3], &txt).expect("write model");
            println!("wrote {} (min_count {chosen}, {} windows, {} bytes)", args[3], m.len(), txt.len());
        }
        "words" => {
            let g = G2p::new();
            for line in stdin.lock().lines() {
                let w = line.unwrap();
                let w = w.trim();
                if !w.is_empty() {
                    writeln!(out, "{w}\t{}", g.word(w)).unwrap();
                }
            }
        }
        "sents" => {
            let g = G2p::new();
            for line in stdin.lock().lines() {
                let l = line.unwrap();
                writeln!(out, "{}", g.sentences(&l).join("\t")).unwrap();
            }
        }
        "ids" => {
            let map = text_to_phonemes::phoneme_id_map_from_config(&std::fs::read_to_string(&args[2]).expect("read config"))
                .expect("parse config");
            for line in stdin.lock().lines() {
                let ids = text_to_phonemes::to_ids(&line.unwrap(), &map);
                writeln!(out, "{}", serde_json::to_string(&ids).unwrap()).unwrap();
            }
        }
        "stats" => {
            let t0 = Instant::now();
            let g = G2p::new();
            println!(
                "lexicon entries {}  oov windows {}  G2p::new {:.1} ms",
                g.lexicon_len(),
                g.oov_len(),
                t0.elapsed().as_secs_f64() * 1e3
            );
            let t0 = Instant::now();
            let text = "Hello there! I'm Lindii, and I'll help you build a game today. It costs $3.50, about 25% off.";
            let n = 200;
            for _ in 0..n {
                std::hint::black_box(g.sentences(text));
            }
            println!("sentences(): {:.3} ms per call ({} chars)", t0.elapsed().as_secs_f64() * 1e3 / n as f64, text.len());
        }
        _ => {
            eprintln!("usage: g2p-tool train|words|sents|ids|stats");
            std::process::exit(2);
        }
    }
}
