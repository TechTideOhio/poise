use crate::pair::{ConsistentOutcome, SplitMix64};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbosityReport {
    pub total_consistent_decisive: usize,
    pub longer_won_count: usize,
    pub shorter_won_count: usize,
    pub equal_length_count: usize,
    pub longer_win_rate: f64,
    pub length_correlation_r: f64,
    pub length_matched_pairs: usize,
    pub length_matched_win_rate_a: f64,
    pub raw_win_rate_a: f64,
    pub length_controlled_win_rate_a: f64,
    pub verbosity_inflation_delta: f64,
    pub lc_bootstrap_95_ci: (f64, f64),
}

fn sigmoid(z: f64) -> f64 {
    if z > 15.0 {
        1.0
    } else if z < -15.0 {
        0.0
    } else {
        1.0 / (1.0 + (-z).exp())
    }
}

pub fn fit_logistic_lc_win_rate(samples: &[(f64, f64)]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.5, 0.0);
    }

    let mut b0 = 0.0;
    let mut b1 = 0.0;

    for _ in 0..25 {
        let mut g0 = 0.0;
        let mut g1 = 0.0;
        let mut h00 = 1e-4;
        let mut h01 = 0.0;
        let mut h11 = 1e-4;

        for &(x, y) in samples {
            let p = sigmoid(b0 + b1 * x);
            let w = (p * (1.0 - p)).max(1e-5);
            let err = y - p;

            g0 += err;
            g1 += err * x;

            h00 += w;
            h01 += w * x;
            h11 += w * x * x;
        }

        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-9 {
            break;
        }

        let db0 = (h11 * g0 - h01 * g1) / det;
        let db1 = (-h01 * g0 + h00 * g1) / det;

        b0 += db0;
        b1 += db1;

        if db0.abs() < 1e-6 && db1.abs() < 1e-6 {
            break;
        }
    }

    let lc_win_rate = sigmoid(b0);
    (lc_win_rate, b1)
}

pub fn analyze_verbosity(
    outcomes_and_lengths: &[(ConsistentOutcome, usize, usize)],
    matched_tau: f64,
    bootstrap_iterations: usize,
    seed: u64,
) -> VerbosityReport {
    let mut decisive_samples = Vec::new();
    let mut longer_won = 0;
    let mut shorter_won = 0;
    let mut equal_len = 0;
    let mut matched_a_wins = 0.0;
    let mut matched_total = 0;

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut sum_xy = 0.0;
    let mut a_wins = 0.0;

    for &(outcome, len_a, len_b) in outcomes_and_lengths {
        if !outcome.is_consistent_decisive() {
            continue;
        }

        let y = match outcome {
            ConsistentOutcome::ConsistentWinA => 1.0,
            ConsistentOutcome::ConsistentWinB => 0.0,
            _ => continue,
        };

        if y > 0.5 {
            a_wins += 1.0;
        }

        let denom = (len_a + len_b).max(1) as f64;
        let x = (len_a as f64 - len_b as f64) / denom;

        decisive_samples.push((x, y));

        if len_a > len_b {
            if y > 0.5 {
                longer_won += 1;
            } else {
                shorter_won += 1;
            }
        } else if len_b > len_a {
            if y < 0.5 {
                longer_won += 1;
            } else {
                shorter_won += 1;
            }
        } else {
            equal_len += 1;
        }

        if x.abs() <= matched_tau {
            matched_total += 1;
            matched_a_wins += y;
        }

        let score_signed = if y > 0.5 { 1.0 } else { -1.0 };
        sum_x += x;
        sum_y += score_signed;
        sum_x2 += x * x;
        sum_y2 += score_signed * score_signed;
        sum_xy += x * score_signed;
    }

    let n = decisive_samples.len();
    let longer_win_rate = if longer_won + shorter_won > 0 {
        longer_won as f64 / (longer_won + shorter_won) as f64
    } else {
        0.50
    };

    let length_correlation_r = if n >= 2 {
        let nf = n as f64;
        let num = nf * sum_xy - sum_x * sum_y;
        let den = ((nf * sum_x2 - sum_x * sum_x) * (nf * sum_y2 - sum_y * sum_y)).sqrt();
        if den.abs() > 1e-9 {
            (num / den).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let length_matched_win_rate_a = if matched_total > 0 {
        matched_a_wins / matched_total as f64
    } else {
        0.50
    };

    let raw_win_rate_a = if n > 0 { a_wins / n as f64 } else { 0.50 };

    let (lc_win_rate, _beta_len) = fit_logistic_lc_win_rate(&decisive_samples);
    let verbosity_inflation_delta = raw_win_rate_a - lc_win_rate;

    let mut prng = SplitMix64::new(seed);
    let mut bootstrap_estimates = Vec::with_capacity(bootstrap_iterations);

    if n >= 4 && bootstrap_iterations > 0 {
        let mut sample_buf = vec![(0.0, 0.0); n];
        for _ in 0..bootstrap_iterations {
            for item in &mut sample_buf {
                let idx = (prng.next_u64() as usize) % n;
                *item = decisive_samples[idx];
            }
            let (boot_lc, _) = fit_logistic_lc_win_rate(&sample_buf);
            bootstrap_estimates.push(boot_lc);
        }
        bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    let lc_bootstrap_95_ci = if !bootstrap_estimates.is_empty() {
        let low_idx = (bootstrap_estimates.len() as f64 * 0.025) as usize;
        let high_idx = ((bootstrap_estimates.len() as f64 * 0.975) as usize)
            .min(bootstrap_estimates.len() - 1);
        (bootstrap_estimates[low_idx], bootstrap_estimates[high_idx])
    } else {
        (lc_win_rate, lc_win_rate)
    };

    VerbosityReport {
        total_consistent_decisive: n,
        longer_won_count: longer_won,
        shorter_won_count: shorter_won,
        equal_length_count: equal_len,
        longer_win_rate,
        length_correlation_r,
        length_matched_pairs: matched_total,
        length_matched_win_rate_a,
        raw_win_rate_a,
        length_controlled_win_rate_a: lc_win_rate,
        verbosity_inflation_delta,
        lc_bootstrap_95_ci,
    }
}
