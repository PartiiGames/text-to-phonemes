// text-to-phonemes integration tests. Each is CPU-only, bounded (no loops over input size beyond the embedded data) and
// prints what it checked, so a farm log shows progress.
use text_to_phonemes::{G2p, phoneme_id_map_from_config, piper_phonemes, to_ids};

/// The phoneme_id_map keys of en_US-hifi_tts_92-medium (166 entries, the same as hfc_female), as listed by the farm
/// job (tools/explore.sh): every symbol text-to-phonemes emits must be one of these.
const HIFI_TTS_92_SYMBOLS: &str = " !\"#$'(),-.0123456789:;?X^_abcdefghijklmnopqrstuvwxyzæçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɝɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢʦʰʲʷˈˌːˑ˞ˤ̧̝̩̪̯̺̻̃̊βεθχᵻ↑↓ⱱ";

fn g2p() -> G2p {
    let g = G2p::new();
    eprintln!("lexicon {} entries, oov {} windows", g.lexicon_len(), g.oov_len());
    g
}

#[test]
fn lexicon_words_match_espeak_style() {
    let g = g2p();
    assert!(g.lexicon_len() > 100_000, "lexicon not embedded");
    // espeak-ng en-us forms (Piper's phonemizer), see tools/compare.py
    for (w, want) in [
        ("hello", "həlˈoʊ"),
        ("world", "wˈɜːld"),
        ("water", "wˈɔːɾɚ"),
        ("better", "bˈɛɾɚ"),
        ("city", "sˈɪɾi"),
        ("party", "pˈɑːɹɾi"),
        ("winter", "wˈɪntɚ"),
        ("ladder", "lˈædɚ"),
        ("button", "bˈʌʔn̩"),
        ("about", "ɐbˈaʊt"),
        ("roses", "ɹˈoʊzᵻz"),
        ("remember", "ɹᵻmˈɛmbɚ"),
        ("being", "bˈiːʲɪŋ"),
        ("more", "mˈɔːɹ"),
        ("hurry", "hˈɜːɹi"),
        ("fire", "fˈaɪɚ"),
        ("information", "ˌɪnfɚmˈeɪʃən"),
        ("judge", "dʒˈʌdʒ"),
        ("cat", "kˈæt"),
    ] {
        let got = g.word(w);
        eprintln!("{w}: {got}");
        assert_eq!(got, want, "{w}");
    }
}

#[test]
fn sentences_split_and_keep_punctuation() {
    let g = g2p();
    let s = g.sentences("Hello, world. How are you today? Great; thanks!");
    eprintln!("{s:?}");
    assert_eq!(s.len(), 3);
    assert_eq!(s[0], "həlˈoʊ, wˈɜːld.");
    assert!(s[1].ends_with('?'));
    assert!(s[2].contains("; ") && s[2].ends_with('!'));
    assert_eq!(g.sentences("The cat sat on the mat."), ["ðə kˈæt sˈæt ɔnðə mˈæt."]);
    assert!(g.sentences("").is_empty());
}

#[test]
fn numbers_and_abbreviations_are_spoken() {
    let g = g2p();
    let s = g.sentences("Dr. Smith paid $3.50 at 3:30 pm.");
    eprintln!("{s:?}");
    assert_eq!(s.len(), 1);
    assert!(s[0].starts_with("dˈɑːktɚ smˈɪθ"));
    assert!(s[0].contains("θɹˈiː dˈɑːlɚz"));
}

#[test]
fn oov_words_get_one_primary_stress() {
    let g = g2p();
    for w in ["blorfing", "zyxtrom", "partiiverse", "lindii", "snorkeling", "quizzaciously"] {
        let ph = g.oov_arpabet(w);
        let ipa = g.word(w);
        eprintln!("{w}: {ipa}");
        assert!(!ph.is_empty(), "{w}");
        assert_eq!(ipa.matches('ˈ').count(), 1, "{w}: {ipa}");
    }
}

#[test]
fn every_symbol_is_in_the_voice_map() {
    let g = g2p();
    let allowed: std::collections::HashSet<char> = HIFI_TTS_92_SYMBOLS.chars().collect();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/sentences.txt")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        for s in g.sentences(line) {
            for c in s.chars() {
                assert!(allowed.contains(&c), "{c:?} (U+{:04X}) in {s}", c as u32);
            }
            n += 1;
        }
    }
    eprintln!("{n} sentences checked");
    assert!(n >= 190);
}

#[test]
fn ids_follow_piper() {
    let cfg = r#"{"phoneme_id_map":{"_":[0],"^":[1],"$":[2]," ":[3],",":[8],".":[10],"h":[20],"l":[24],"o":[27],
        "ə":[59],"ˈ":[120],"ʊ":[100],"aɪ":[161]}}"#;
    let map = phoneme_id_map_from_config(cfg).unwrap();
    // ^ _ h _ ə _ l _ ˈ _ o _ ʊ _ . _ $
    assert_eq!(to_ids("həlˈoʊ.", &map), [1, 0, 20, 0, 59, 0, 24, 0, 120, 0, 27, 0, 100, 0, 10, 0, 2]);
    // unknown symbols are skipped
    assert_eq!(to_ids("hq", &map), [1, 0, 20, 0, 2]);
    // vowel clusters merge only when the config lists them
    let cfg2 = r#"{"phoneme_id_map":{"_":[0],"^":[1],"$":[2],"a":[14],"ɪ":[74],"aɪ":[161]},"vowel_clusters":[["a","ɪ"]]}"#;
    let map2 = phoneme_id_map_from_config(cfg2).unwrap();
    assert_eq!(piper_phonemes("aɪ", &map2), ["aɪ"]);
    assert_eq!(to_ids("aɪ", &map2), [1, 0, 161, 0, 2]);
    assert_eq!(to_ids("aɪ", &map), [1, 0, 2]); // "a", "ɪ" alone are not in this map
}
