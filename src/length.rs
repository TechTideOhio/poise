use serde::{Deserialize, Serialize};
use std::borrow::Cow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLengthStats {
    pub char_count: usize,
    pub byte_count: usize,
    pub line_count: usize,
    pub est_tokens: usize,
}

impl TextLengthStats {
    pub fn compute(text: &str) -> Self {
        let char_count = text.chars().count();
        let byte_count = text.len();
        let line_count = text.lines().count();
        let est_tokens = (char_count + 3) / 4;
        Self {
            char_count,
            byte_count,
            line_count,
            est_tokens,
        }
    }
}

pub fn strip_fluff_boilerplate(text: &str) -> Cow<'_, str> {
    let lower = text.to_lowercase();
    let prefixes = [
        "certainly! here is",
        "certainly, here is",
        "here is a comprehensive",
        "sure, here is",
        "of course! here is",
        "i would be happy to help",
    ];

    for prefix in &prefixes {
        if lower.starts_with(prefix) {
            if let Some(newline_pos) = text.find('\n') {
                let remainder = text[newline_pos + 1..].trim_start();
                return Cow::Owned(remainder.to_string());
            }
        }
    }

    Cow::Borrowed(text)
}

pub fn cap_text_at_budget(text: &str, max_tokens: usize, max_chars: usize) -> (Cow<'_, str>, bool) {
    let stats = TextLengthStats::compute(text);
    if stats.est_tokens <= max_tokens && stats.char_count <= max_chars {
        return (Cow::Borrowed(text), false);
    }

    let target_chars = max_chars.min(max_tokens * 4);
    let mut boundary = 0;
    for (idx, ch) in text.char_indices() {
        if idx >= target_chars {
            break;
        }
        if ch == '.' || ch == '\n' || ch == ';' {
            boundary = idx + ch.len_utf8();
        }
    }

    if boundary == 0 || boundary < target_chars / 2 {
        let mut char_count = 0;
        boundary = 0;
        for (idx, ch) in text.char_indices() {
            if char_count >= target_chars {
                break;
            }
            boundary = idx + ch.len_utf8();
            char_count += 1;
        }
    }

    let truncated = format!("{} [truncated at budget]", text[..boundary].trim_end());
    (Cow::Owned(truncated), true)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedPairStats {
    pub a_orig_stats: TextLengthStats,
    pub b_orig_stats: TextLengthStats,
    pub a_post_stats: TextLengthStats,
    pub b_post_stats: TextLengthStats,
    pub a_was_truncated: bool,
    pub b_was_truncated: bool,
    pub token_delta_ratio: f64,
}

pub fn normalize_and_cap_pair(
    response_a: &str,
    response_b: &str,
    max_tokens: usize,
    max_chars: usize,
    strip_fluff: bool,
) -> (String, String, NormalizedPairStats) {
    let a_orig = TextLengthStats::compute(response_a);
    let b_orig = TextLengthStats::compute(response_b);

    let clean_a = if strip_fluff {
        strip_fluff_boilerplate(response_a)
    } else {
        Cow::Borrowed(response_a)
    };

    let clean_b = if strip_fluff {
        strip_fluff_boilerplate(response_b)
    } else {
        Cow::Borrowed(response_b)
    };

    let (capped_a, a_trunc) = cap_text_at_budget(&clean_a, max_tokens, max_chars);
    let (capped_b, b_trunc) = cap_text_at_budget(&clean_b, max_tokens, max_chars);

    let a_post = TextLengthStats::compute(&capped_a);
    let b_post = TextLengthStats::compute(&capped_b);

    let denom = (a_post.est_tokens + b_post.est_tokens).max(1) as f64;
    let token_delta_ratio = (a_post.est_tokens as f64 - b_post.est_tokens as f64) / denom;

    let stats = NormalizedPairStats {
        a_orig_stats: a_orig,
        b_orig_stats: b_orig,
        a_post_stats: a_post,
        b_post_stats: b_post,
        a_was_truncated: a_trunc,
        b_was_truncated: b_trunc,
        token_delta_ratio,
    };

    (capped_a.into_owned(), capped_b.into_owned(), stats)
}
