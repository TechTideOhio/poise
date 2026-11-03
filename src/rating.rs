use crate::pair::{adjudicate_pair, ConsistentOutcome, PairwiseComparisonRecord, SplitMix64};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateTournamentRating {
    pub candidate_id: String,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    pub naive_win_rate: f64,
    pub consistent_win_rate: f64,
    pub bradley_terry_strength: f64,
    pub elo_rating: f64,
    pub elo_bootstrap_95_ci: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentReport {
    pub total_pairs: usize,
    pub consistent_pairs: usize,
    pub quarantined_flips: usize,
    pub ratings: Vec<CandidateTournamentRating>,
}

pub fn solve_bradley_terry(
    candidates: &[String],
    wins_matrix: &[Vec<f64>], // [i][j] = wins of candidate i over j
    iterations: usize,
) -> Vec<f64> {
    let m = candidates.len();
    if m <= 1 {
        return vec![1.0; m];
    }

    let mut p = vec![1.0 / m as f64; m];

    for _ in 0..iterations {
        let mut p_next = vec![0.0; m];

        for i in 0..m {
            let mut w_i = 0.0;
            let mut denom_sum = 0.0;

            for j in 0..m {
                if i == j {
                    continue;
                }
                let w_ij = wins_matrix[i][j];
                let w_ji = wins_matrix[j][i];
                let total_ij = w_ij + w_ji;

                w_i += w_ij;
                if total_ij > 0.0 {
                    denom_sum += total_ij / (p[i] + p[j]);
                }
            }

            if denom_sum > 1e-9 {
                p_next[i] = (w_i + 1e-4) / denom_sum;
            } else {
                p_next[i] = p[i];
            }
        }

        let sum: f64 = p_next.iter().sum();
        if sum > 1e-9 {
            for v in &mut p_next {
                *v /= sum;
            }
        }
        p = p_next;
    }

    p
}

pub fn rank_tournament(
    records: &[PairwiseComparisonRecord],
    bootstrap_iterations: usize,
    seed: u64,
) -> TournamentReport {
    let mut candidate_set = BTreeMap::new();
    for rec in records {
        candidate_set.insert(rec.forward_pass.slot1_candidate.clone(), 0);
        candidate_set.insert(rec.forward_pass.slot2_candidate.clone(), 0);
    }

    let candidates: Vec<String> = candidate_set.into_keys().collect();
    let m = candidates.len();
    let index_map: BTreeMap<String, usize> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), i))
        .collect();

    let mut wins_matrix = vec![vec![0.0; m]; m];
    let mut naive_wins = vec![0.0; m];
    let mut naive_totals = vec![0.0; m];

    let mut wins = vec![0_usize; m];
    let mut losses = vec![0_usize; m];
    let mut ties = vec![0_usize; m];

    let mut consistent_count = 0;
    let mut quarantined_flips = 0;

    for rec in records {
        let a_str = &rec.forward_pass.slot1_candidate;
        let b_str = &rec.forward_pass.slot2_candidate;
        let idx_a = index_map[a_str];
        let idx_b = index_map[b_str];

        naive_totals[idx_a] += 1.0;
        naive_totals[idx_b] += 1.0;
        if rec.forward_pass.verdict == crate::pair::SlotVerdict::Slot1 {
            naive_wins[idx_a] += 1.0;
        } else if rec.forward_pass.verdict == crate::pair::SlotVerdict::Slot2 {
            naive_wins[idx_b] += 1.0;
        }

        let outcome = adjudicate_pair(rec, a_str, b_str, 0)
            .unwrap_or(ConsistentOutcome::InconsistentPartialFlip);

        match outcome {
            ConsistentOutcome::ConsistentWinA => {
                wins_matrix[idx_a][idx_b] += 1.0;
                wins[idx_a] += 1;
                losses[idx_b] += 1;
                consistent_count += 1;
            }
            ConsistentOutcome::ConsistentWinB => {
                wins_matrix[idx_b][idx_a] += 1.0;
                wins[idx_b] += 1;
                losses[idx_a] += 1;
                consistent_count += 1;
            }
            ConsistentOutcome::ConsistentTie => {
                wins_matrix[idx_a][idx_b] += 0.5;
                wins_matrix[idx_b][idx_a] += 0.5;
                ties[idx_a] += 1;
                ties[idx_b] += 1;
                consistent_count += 1;
            }
            _ => {
                quarantined_flips += 1;
            }
        }
    }

    let bt_strengths = solve_bradley_terry(&candidates, &wins_matrix, 50);

    let mut ratings = Vec::with_capacity(m);
    for (i, c) in candidates.iter().enumerate() {
        let n_c = (wins[i] + losses[i] + ties[i]).max(1) as f64;
        let consistent_wr = (wins[i] as f64 + 0.5 * ties[i] as f64) / n_c;
        let naive_wr = if naive_totals[i] > 0.0 {
            naive_wins[i] / naive_totals[i]
        } else {
            0.50
        };

        let beta_i = bt_strengths[i].ln();
        let elo = 1500.0 + (400.0 / std::f64::consts::LN_10) * beta_i;

        ratings.push(CandidateTournamentRating {
            candidate_id: c.clone(),
            wins: wins[i],
            losses: losses[i],
            ties: ties[i],
            naive_win_rate: naive_wr,
            consistent_win_rate: consistent_wr,
            bradley_terry_strength: bt_strengths[i],
            elo_rating: elo,
            elo_bootstrap_95_ci: (elo - 15.0, elo + 15.0),
        });
    }

    let mut prng = SplitMix64::new(seed);
    if records.len() >= 6 && bootstrap_iterations > 0 {
        let mut boot_elos: Vec<Vec<f64>> = vec![Vec::with_capacity(bootstrap_iterations); m];

        for _ in 0..bootstrap_iterations {
            let mut boot_wins = vec![vec![0.0; m]; m];
            for _ in 0..records.len() {
                let r_idx = (prng.next_u64() as usize) % records.len();
                let rec = &records[r_idx];
                let a_str = &rec.forward_pass.slot1_candidate;
                let b_str = &rec.forward_pass.slot2_candidate;
                let idx_a = index_map[a_str];
                let idx_b = index_map[b_str];

                if let Ok(outcome) = adjudicate_pair(rec, a_str, b_str, 0) {
                    match outcome {
                        ConsistentOutcome::ConsistentWinA => boot_wins[idx_a][idx_b] += 1.0,
                        ConsistentOutcome::ConsistentWinB => boot_wins[idx_b][idx_a] += 1.0,
                        ConsistentOutcome::ConsistentTie => {
                            boot_wins[idx_a][idx_b] += 0.5;
                            boot_wins[idx_b][idx_a] += 0.5;
                        }
                        _ => {}
                    }
                }
            }

            let boot_bt = solve_bradley_terry(&candidates, &boot_wins, 25);
            for i in 0..m {
                let boot_elo = 1500.0 + (400.0 / std::f64::consts::LN_10) * boot_bt[i].ln();
                boot_elos[i].push(boot_elo);
            }
        }

        for i in 0..m {
            boot_elos[i].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let low_idx = (bootstrap_iterations as f64 * 0.025) as usize;
            let high_idx =
                ((bootstrap_iterations as f64 * 0.975) as usize).min(bootstrap_iterations - 1);
            ratings[i].elo_bootstrap_95_ci = (boot_elos[i][low_idx], boot_elos[i][high_idx]);
        }
    }

    ratings.sort_by(|a, b| {
        b.elo_rating
            .partial_cmp(&a.elo_rating)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    TournamentReport {
        total_pairs: records.len(),
        consistent_pairs: consistent_count,
        quarantined_flips,
        ratings,
    }
}
