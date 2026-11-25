//! Text normalization: sentences of lowercase words and prosody punctuation (. , ! ? ; :), the marks Piper's
//! phonemizer keeps. Numbers, money, times, percents, ordinals, years, common abbreviations, symbols and acronyms
//! become words.

use crate::numbers::{cardinal, digits, ordinal, year};

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Word(String),
    /// One of . , ! ? ; :
    Punct(char),
    /// A letter read by name (acronyms, initials, a.m.)
    Letter(char),
}

/// Abbreviations (lowercase, without the final period) that never end a sentence.
const TITLES: &[(&str, &str)] = &[
    ("mr", "mister"),
    ("mrs", "missus"),
    ("ms", "miz"),
    ("dr", "doctor"),
    ("prof", "professor"),
    ("sr", "senior"),
    ("jr", "junior"),
    ("st", "saint"),
    ("mt", "mount"),
    ("gen", "general"),
    ("gov", "governor"),
    ("sen", "senator"),
    ("rep", "representative"),
    ("capt", "captain"),
    ("lt", "lieutenant"),
    ("col", "colonel"),
    ("sgt", "sergeant"),
    ("rev", "reverend"),
    ("vs", "versus"),
    ("no", "number"),
    ("vol", "volume"),
    ("fig", "figure"),
    ("approx", "approximately"),
    ("dept", "department"),
    ("est", "established"),
];
/// Abbreviations that may also end a sentence (when the next word is capitalized).
const ENDERS: &[(&str, &str)] = &[
    ("etc", "et cetera"),
    ("ave", "avenue"),
    ("blvd", "boulevard"),
    ("rd", "road"),
    ("inc", "incorporated"),
    ("ltd", "limited"),
    ("co", "company"),
    ("corp", "corporation"),
    ("e.g", "for example"),
    ("i.e", "that is"),
    ("a.m", "a m"),
    ("p.m", "p m"),
    ("u.s", "u s"),
    ("u.k", "u k"),
    ("jan", "january"),
    ("feb", "february"),
    ("mar", "march"),
    ("apr", "april"),
    ("aug", "august"),
    ("sept", "september"),
    ("sep", "september"),
    ("oct", "october"),
    ("nov", "november"),
    ("dec", "december"),
    ("mon", "monday"),
    ("tue", "tuesday"),
    ("wed", "wednesday"),
    ("thu", "thursday"),
    ("fri", "friday"),
    ("sat", "saturday"),
    ("sun", "sunday"),
    ("min", "minutes"),
    ("hr", "hours"),
    ("hrs", "hours"),
    ("ft", "feet"),
    ("lbs", "pounds"),
    ("oz", "ounces"),
    ("km", "kilometers"),
    ("kg", "kilograms"),
];

fn lookup(table: &[(&str, &'static str)], k: &str) -> Option<&'static str> {
    table.iter().find(|(a, _)| *a == k).map(|(_, b)| *b)
}

fn push_words(out: &mut Vec<Tok>, s: &str) {
    for w in s.split_whitespace() {
        out.push(Tok::Word(w.to_string()));
    }
}

fn is_word(t: &Tok) -> bool {
    matches!(t, Tok::Word(_) | Tok::Letter(_))
}

/// An abbreviation's expansion; "a m" style (all single letters) is read letter by letter.
fn push_expansion(out: &mut Vec<Tok>, s: &str) {
    if s.split_whitespace().all(|w| w.chars().count() == 1) {
        out.extend(s.chars().filter(|c| !c.is_whitespace()).map(Tok::Letter));
    } else {
        push_words(out, s);
    }
}

fn push_all(out: &mut Vec<Tok>, ws: Vec<String>) {
    for w in ws {
        out.push(Tok::Word(w));
    }
}

/// "1,234.5" -> words. `int` may contain commas.
fn number_words(int: &str, frac: Option<&str>, allow_year: bool) -> Vec<String> {
    let clean: String = int.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut v = if clean.is_empty() {
        Vec::new()
    } else if clean.len() > 15 {
        digits(&clean)
    } else if clean.len() > 1 && clean.starts_with('0') {
        digits(&clean)
    } else {
        let n: u64 = clean.parse().unwrap_or(0);
        if allow_year && frac.is_none() && !int.contains(',') && clean.len() == 4 && (1100..=2099).contains(&n) {
            year(n)
        } else {
            cardinal(n)
        }
    };
    if let Some(f) = frac {
        if v.is_empty() {
            v.push("zero".into());
        }
        v.push("point".into());
        v.extend(digits(f));
    }
    v
}

/// Split "12.50" into ("12", Some("50")).
fn split_decimal(s: &str) -> (&str, Option<&str>) {
    match s.split_once('.') {
        Some((a, b)) if !b.is_empty() && b.chars().all(|c| c.is_ascii_digit()) => (a, Some(b)),
        _ => (s.trim_end_matches('.'), None),
    }
}

fn is_num(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_ascii_digit())
        && s.chars().all(|c| c.is_ascii_digit() || c == ',' || c == '.')
}

fn money(amount: &str, unit: (&str, &str), sub: (&str, &str), out: &mut Vec<Tok>) {
    let (int, frac) = split_decimal(amount);
    let n: u64 = int.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
    let cents: u64 = match frac {
        Some(f) if f.len() == 2 => f.parse().unwrap_or(0),
        Some(f) => {
            // "$1.5" style: a plain decimal
            push_all(out, number_words(int, Some(f), false));
            push_words(out, unit.1);
            return;
        }
        None => 0,
    };
    if n > 0 || cents == 0 {
        push_all(out, number_words(int, None, false));
        push_words(out, if n == 1 { unit.0 } else { unit.1 });
    }
    if cents > 0 {
        if n > 0 {
            out.push(Tok::Word("and".into()));
        }
        push_all(out, cardinal(cents));
        push_words(out, if cents == 1 { sub.0 } else { sub.1 });
    }
}

/// Words for one token core (no leading/trailing punctuation). `known` says whether a word is in the lexicon.
fn expand(core: &str, known: &dyn Fn(&str) -> bool, out: &mut Vec<Tok>) {
    if core.is_empty() {
        return;
    }
    let lower = core.to_lowercase();
    // currency
    for (sym, unit, sub) in [
        ('$', ("dollar", "dollars"), ("cent", "cents")),
        ('£', ("pound", "pounds"), ("penny", "pence")),
        ('€', ("euro", "euros"), ("cent", "cents")),
    ] {
        if let Some(rest) = core.strip_prefix(sym) {
            let (amt, scale) = match rest.find(|c: char| c.is_ascii_alphabetic()) {
                Some(i) => (&rest[..i], Some(rest[i..].to_lowercase())),
                None => (rest, None),
            };
            if is_num(amt) {
                if let Some(sc) = scale {
                    // $5m, $2bn, $3k
                    let (int, frac) = split_decimal(amt);
                    push_all(out, number_words(int, frac, false));
                    let w = match sc.as_str() {
                        "k" => "thousand",
                        "m" | "mn" => "million",
                        "b" | "bn" => "billion",
                        "t" | "tn" => "trillion",
                        other => other,
                    };
                    push_words(out, w);
                    push_words(out, unit.1);
                } else {
                    money(amt, unit, sub, out);
                }
                return;
            }
        }
    }
    // percent
    if let Some(n) = core.strip_suffix('%') {
        if is_num(n) {
            let (int, frac) = split_decimal(n);
            push_all(out, number_words(int, frac, false));
            out.push(Tok::Word("percent".into()));
            return;
        }
    }
    // time h:mm (optionally am/pm glued on)
    if let Some((h, m)) = core.split_once(':') {
        let (m, ampm) = match m.find(|c: char| c.is_ascii_alphabetic()) {
            Some(i) => (&m[..i], Some(m[i..].to_lowercase())),
            None => (m, None),
        };
        if !h.is_empty() && h.len() <= 2 && h.chars().all(|c| c.is_ascii_digit()) && m.len() == 2 && m.chars().all(|c| c.is_ascii_digit()) {
            let (hn, mn): (u64, u64) = (h.parse().unwrap_or(0), m.parse().unwrap_or(0));
            push_all(out, cardinal(hn));
            match mn {
                0 if ampm.is_none() => out.push(Tok::Word("o'clock".into())),
                0 => {}
                1..=9 => {
                    out.push(Tok::Word("oh".into()));
                    push_all(out, cardinal(mn));
                }
                _ => push_all(out, cardinal(mn)),
            }
            if let Some(ap) = ampm {
                for c in ap.chars().filter(|c| c.is_ascii_alphabetic()) {
                    out.push(Tok::Letter(c));
                }
            }
            return;
        }
    }
    // ordinals 1st 2nd 3rd 4th
    for suf in ["st", "nd", "rd", "th"] {
        if let Some(n) = lower.strip_suffix(suf) {
            if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit() || c == ',') {
                let v: u64 = n.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
                push_all(out, ordinal(v));
                return;
            }
        }
    }
    // decades: 1990s, 80s, '90s
    if let Some(n) = lower.trim_start_matches('\'').strip_suffix('s') {
        if (n.len() == 4 || n.len() == 2) && n.chars().all(|c| c.is_ascii_digit()) && n.ends_with('0') {
            let mut v = if n.len() == 4 { year(n.parse().unwrap_or(0)) } else { cardinal(n.parse().unwrap_or(0)) };
            if let Some(last) = v.last_mut() {
                *last = if last.ends_with('y') { format!("{}ies", &last[..last.len() - 1]) } else { format!("{last}s") };
            }
            push_all(out, v);
            return;
        }
    }
    // plain numbers, negative numbers
    if is_num(core) {
        let (int, frac) = split_decimal(core);
        push_all(out, number_words(int, frac, true));
        return;
    }
    if let Some(n) = core.strip_prefix('-').or_else(|| core.strip_prefix('−')) {
        if is_num(n) {
            out.push(Tok::Word("minus".into()));
            let (int, frac) = split_decimal(n);
            push_all(out, number_words(int, frac, false));
            return;
        }
    }
    // dotted acronyms: U.S.A, e.g (period stripped already)
    if core.contains('.') && core.split('.').all(|p| p.chars().count() == 1 && p.chars().all(char::is_alphabetic)) {
        for p in core.split('.') {
            out.extend(p.to_lowercase().chars().map(Tok::Letter));
        }
        return;
    }
    // symbols and joiners inside the token
    if let Some(i) = core.find(['&', '+', '=', '@', '/', '#', '-', '–', '_', '.', '°', '*', ':', '"', '(', ')']) {
        let c = core[i..].chars().next().unwrap();
        let (a, b) = (&core[..i], &core[i + c.len_utf8()..]);
        expand(a, known, out);
        match c {
            '&' => push_words(out, "and"),
            '+' => push_words(out, "plus"),
            '=' => push_words(out, "equals"),
            '@' => push_words(out, "at"),
            '/' => {
                if !(a.is_empty() || b.is_empty()) {
                    push_words(out, if is_num(a) && is_num(b) { "over" } else { "slash" })
                }
            }
            '#' => push_words(out, if b.starts_with(|c: char| c.is_ascii_digit()) { "number" } else { "hash" }),
            '°' => push_words(out, "degrees"),
            '.' if is_num(a) || is_num(b) => push_words(out, "point"),
            _ => {}
        }
        expand(b, known, out);
        return;
    }
    // letters and digits glued together: mp3, 4k, b2b
    let has_digit = core.chars().any(|c| c.is_ascii_digit());
    let has_alpha = core.chars().any(|c| c.is_alphabetic());
    if has_digit && has_alpha {
        let mut run = String::new();
        let mut run_digit = false;
        for c in core.chars() {
            let d = c.is_ascii_digit();
            if !run.is_empty() && d != run_digit {
                expand(&run, known, out);
                run.clear();
            }
            run_digit = d;
            run.push(c);
        }
        expand(&run, known, out);
        return;
    }
    if !has_alpha {
        // other symbols alone
        match core {
            "&" => push_words(out, "and"),
            "+" => push_words(out, "plus"),
            "=" => push_words(out, "equals"),
            "@" => push_words(out, "at"),
            "%" => push_words(out, "percent"),
            "$" => push_words(out, "dollars"),
            _ => {}
        }
        return;
    }
    let word: String = lower.trim_matches('\'').to_string();
    if word.is_empty() {
        return;
    }
    // acronyms: all caps, 2-5 letters, unknown -> letters
    let letters: Vec<char> = core.chars().filter(|c| c.is_alphabetic()).collect();
    let all_caps = letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase());
    if all_caps && letters.len() <= 5 && !known(&word) {
        for c in letters {
            out.extend(c.to_lowercase().map(Tok::Letter));
        }
        return;
    }
    out.push(Tok::Word(word));
}

fn clause_mark(c: char) -> Option<char> {
    match c {
        '.' | '…' => Some('.'),
        '!' => Some('!'),
        '?' => Some('?'),
        ',' => Some(','),
        ';' => Some(';'),
        ':' => Some(':'),
        _ => None,
    }
}

fn is_wrapper(c: char) -> bool {
    matches!(c, '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '“' | '”' | '‘' | '’' | '«' | '»' | '*' | '_')
}

fn end_sentence(cur: &mut Vec<Tok>, sentences: &mut Vec<Vec<Tok>>) {
    if cur.iter().any(|t| is_word(t)) {
        sentences.push(std::mem::take(cur));
    } else {
        cur.clear();
    }
}

/// Normalize text into sentences. Each sentence ends with its . ! or ? (a final sentence without one gets none).
pub fn normalize(text: &str, known: &dyn Fn(&str) -> bool) -> Vec<Vec<Tok>> {
    let text = text.replace(['’', '‘', 'ʼ'], "'").replace(['“', '”'], "\"").replace(['—', '–'], " , ");
    let raw: Vec<&str> = text.split_whitespace().collect();
    let mut sentences: Vec<Vec<Tok>> = Vec::new();
    let mut cur: Vec<Tok> = Vec::new();
    for (ti, tok) in raw.iter().enumerate() {
        let next_cap = raw
            .get(ti + 1)
            .and_then(|t| t.trim_start_matches(is_wrapper).chars().next())
            .is_none_or(|c| c.is_uppercase() || c.is_ascii_digit());
        let mut s: &str = tok.trim_start_matches(is_wrapper);
        // a leading clause mark ("-" alone, ",") attaches to what came before
        while let Some(c) = s.chars().next() {
            if let Some(m) = clause_mark(c) {
                if m == ',' && cur.last().is_some_and(|t| is_word(t)) {
                    cur.push(Tok::Punct(','));
                }
                s = &s[c.len_utf8()..];
            } else if c == '-' && s.len() == 1 {
                s = "";
                if cur.last().is_some_and(|t| is_word(t)) {
                    cur.push(Tok::Punct(','));
                }
            } else {
                break;
            }
        }
        // trailing marks and wrappers
        let mut marks: Vec<char> = Vec::new();
        loop {
            let Some(c) = s.chars().last() else { break };
            if is_wrapper(c) {
                s = &s[..s.len() - c.len_utf8()];
            } else if let Some(m) = clause_mark(c) {
                marks.push(m);
                s = &s[..s.len() - c.len_utf8()];
            } else {
                break;
            }
        }
        marks.reverse();
        // a final period may belong to an abbreviation or an initial
        let mut word_done = false;
        if marks.first() == Some(&'.') && !s.is_empty() {
            let low = s.to_lowercase();
            if let Some(full) = lookup(TITLES, &low) {
                let next_is_num = raw.get(ti + 1).is_some_and(|t| t.starts_with(|c: char| c.is_ascii_digit()));
                // "St." after a capitalized name is street; "No." only before a number
                let after_name = cur.last().is_some_and(|t| is_word(t))
                    && ti > 0
                    && raw[ti - 1].trim_start_matches(is_wrapper).starts_with(char::is_uppercase);
                let w = if low == "st" && (after_name || !next_cap) {
                    "street"
                } else if low == "no" && !next_is_num {
                    "no"
                } else {
                    full
                };
                if !(low == "no" && !next_is_num) {
                    push_expansion(&mut cur, w);
                    marks.remove(0);
                    word_done = true;
                }
            } else if let Some(full) = lookup(ENDERS, &low) {
                push_expansion(&mut cur, full);
                marks.remove(0);
                word_done = true;
                if next_cap {
                    marks.insert(0, '.'); // it also ends the sentence
                }
            } else if s.chars().count() == 1 && s.chars().all(|c| c.is_uppercase()) && s != "I" && next_cap && ti + 1 < raw.len() {
                // an initial: "J. K. Rowling"
                cur.extend(s.to_lowercase().chars().map(Tok::Letter));
                marks.remove(0);
                word_done = true;
            }
        }
        // "3:30 pm", "7 am": letters
        let low = s.to_lowercase();
        if !word_done
            && (low == "am" || low == "pm")
            && ti > 0
            && raw[ti - 1].chars().next().is_some_and(|c| c.is_ascii_digit())
        {
            cur.extend(low.chars().map(Tok::Letter));
            word_done = true;
        }
        if !word_done {
            expand(s, known, &mut cur);
        }
        // the strongest mark wins: ? ! . end the sentence, else the first clause mark
        if !marks.is_empty() && cur.last().is_some_and(|t| is_word(t)) {
            let m = if marks.contains(&'?') {
                '?'
            } else if marks.contains(&'!') {
                '!'
            } else {
                marks[0]
            };
            cur.push(Tok::Punct(m));
            if matches!(m, '.' | '!' | '?') {
                end_sentence(&mut cur, &mut sentences);
            }
        }
    }
    end_sentence(&mut cur, &mut sentences);
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(text: &str) -> Vec<String> {
        normalize(text, &|w| w == "nasa")
            .into_iter()
            .map(|s| {
                s.iter()
                    .map(|t| match t {
                        Tok::Word(w) => w.clone(),
                        Tok::Punct(c) | Tok::Letter(c) => c.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    #[test]
    fn sentences_and_marks() {
        assert_eq!(flat("Hello, world. How are you?"), ["hello , world .", "how are you ?"]);
        assert_eq!(flat("Wait; what: no!"), ["wait ; what : no !"]);
        assert_eq!(flat("No end"), ["no end"]);
    }

    #[test]
    fn numbers_money_time() {
        assert_eq!(flat("I paid $3.50 for 25% at 3:30pm."), ["i paid three dollars and fifty cents for twenty five percent at three thirty p m ."]);
        assert_eq!(flat("It was 1990 and 2,500 people came 2nd."), ["it was nineteen ninety and two thousand five hundred people came second ."]);
        assert_eq!(flat("Pi is 3.14, not -2."), ["pi is three point one four , not minus two ."]);
        assert_eq!(flat("At 7:05 or 9:00."), ["at seven oh five or nine o'clock ."]);
    }

    #[test]
    fn abbreviations() {
        assert_eq!(flat("Dr. Smith lives on Elm St. near Mt. Hood."), ["doctor smith lives on elm street near mount hood ."]);
        assert_eq!(flat("St. Louis is big."), ["saint louis is big ."]);
        assert_eq!(flat("Apples, pears, etc. are fruit."), ["apples , pears , et cetera are fruit ."]);
        assert_eq!(flat("The FBI and NASA met."), ["the f b i and nasa met ."]);
        assert_eq!(flat("Rock & roll, well-known."), ["rock and roll , well known ."]);
    }
}
