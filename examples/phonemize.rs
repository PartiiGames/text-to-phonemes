//! Phonemize text from the command line, one IPA line per sentence; with a Piper voice config, also print ids.
//!   cargo run --example phonemize -- "Hello, world. It costs $3.50!" [en_US-hifi_tts_92-medium.onnx.json]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let text = args.first().map(String::as_str).unwrap_or("Hello, world. It costs $3.50!");
    let map = args.get(1).map(|p| {
        let cfg = std::fs::read_to_string(p).expect("read voice config");
        text_to_phonemes::phoneme_id_map_from_config(&cfg).expect("parse voice config")
    });
    let g2p = text_to_phonemes::G2p::new();
    for ipa in g2p.sentences(text) {
        println!("{ipa}");
        if let Some(map) = &map {
            println!("  ids {:?}", text_to_phonemes::to_ids(&ipa, map));
        }
    }
}
