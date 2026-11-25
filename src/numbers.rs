//! Numbers to English words (US style: no "and" after hundred).

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES: [(u64, &str); 4] =
    [(1_000_000_000_000, "trillion"), (1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")];

fn below_thousand(n: u64, out: &mut Vec<String>) {
    let (h, r) = (n / 100, n % 100);
    if h > 0 {
        out.push(ONES[h as usize].into());
        out.push("hundred".into());
    }
    if r > 0 {
        if r < 20 {
            out.push(ONES[r as usize].into());
        } else {
            out.push(TENS[(r / 10) as usize].into());
            if r % 10 > 0 {
                out.push(ONES[(r % 10) as usize].into());
            }
        }
    }
}

/// Cardinal words, e.g. 1234 -> one thousand two hundred thirty four.
pub fn cardinal(n: u64) -> Vec<String> {
    if n == 0 {
        return vec!["zero".into()];
    }
    let mut out = Vec::new();
    let mut rest = n;
    for (v, name) in SCALES {
        if rest >= v {
            if rest / v >= 1000 {
                // beyond trillions: digit by digit
                return digits(&n.to_string());
            }
            below_thousand(rest / v, &mut out);
            out.push(name.into());
            rest %= v;
        }
    }
    below_thousand(rest, &mut out);
    out
}

/// Each digit as a word.
pub fn digits(s: &str) -> Vec<String> {
    s.chars().filter_map(|c| c.to_digit(10)).map(|d| ONES[d as usize].to_string()).collect()
}

/// A year: 1990 -> nineteen ninety, 2005 -> two thousand five, 2024 -> twenty twenty four, 1900 -> nineteen hundred.
pub fn year(n: u64) -> Vec<String> {
    if !(1000..=2999).contains(&n) || (2000..2010).contains(&n) || n % 1000 == 0 {
        return cardinal(n);
    }
    let (hi, lo) = (n / 100, n % 100);
    let mut out = cardinal(hi);
    match lo {
        0 => out.push("hundred".into()),
        1..=9 => {
            out.push("oh".into());
            out.extend(cardinal(lo));
        }
        _ => out.extend(cardinal(lo)),
    }
    out
}

fn ordinal_word(w: &str) -> String {
    match w {
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    }
}

/// Ordinal words, e.g. 21 -> twenty first.
pub fn ordinal(n: u64) -> Vec<String> {
    let mut v = cardinal(n);
    if let Some(last) = v.last_mut() {
        *last = ordinal_word(last);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    fn j(v: Vec<String>) -> String {
        v.join(" ")
    }
    #[test]
    fn numbers() {
        assert_eq!(j(cardinal(0)), "zero");
        assert_eq!(j(cardinal(13)), "thirteen");
        assert_eq!(j(cardinal(1234)), "one thousand two hundred thirty four");
        assert_eq!(j(cardinal(2_000_005)), "two million five");
        assert_eq!(j(year(1990)), "nineteen ninety");
        assert_eq!(j(year(2005)), "two thousand five");
        assert_eq!(j(year(1905)), "nineteen oh five");
        assert_eq!(j(year(2024)), "twenty twenty four");
        assert_eq!(j(ordinal(21)), "twenty first");
        assert_eq!(j(ordinal(40)), "fortieth");
        assert_eq!(j(ordinal(12)), "twelfth");
        assert_eq!(j(ordinal(100)), "one hundredth");
    }
}
