// Builds the embedded data into OUT_DIR:
//   lexicon.bin  from data/cmudict.dict (CMUdict, LICENSE-CMUDICT), the compact front-coded table (src/phoneset.rs)
//   oov.model    copied from data/oov.model (trained only on CMUdict by `g2p-tool train`)
// A missing input gives an empty file (and a warning): the crate still builds, every word goes to the fallback.
#![allow(dead_code)]
include!("src/phoneset.rs");

fn main() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=data/cmudict.dict");
    println!("cargo:rerun-if-changed=data/oov.model");
    println!("cargo:rerun-if-changed=src/phoneset.rs");
    let lex = match std::fs::read_to_string("data/cmudict.dict") {
        Ok(text) => encode_lexicon(&parse_cmudict(&text)),
        Err(_) => {
            println!("cargo:warning=data/cmudict.dict missing: building with an empty lexicon");
            Vec::new()
        }
    };
    std::fs::write(out.join("lexicon.bin"), lex).unwrap();
    let model = std::fs::read("data/oov.model").unwrap_or_else(|_| {
        println!("cargo:warning=data/oov.model missing: building without the OOV model");
        Vec::new()
    });
    std::fs::write(out.join("oov.model"), model).unwrap();
}
