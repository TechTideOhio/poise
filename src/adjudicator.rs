use crate::length::TextLengthStats;
use crate::pair::{adjudicate_pair, ConsistentOutcome, PairError, PairwiseComparisonRecord};
use crate::spec::{CaseSeverity, PairwiseCaseSpec};
use crate::verbosity::{analyze_verbosity, VerbosityReport};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairwiseBiasRegime {
    EquilibriumPoised,
    PrimacyBiasSkewed,
    RecencyBiasSkewed,
    VerbositySkewDistorted,
    LengthBudgetExceeded,
    MissingSwappedPassBlocked,
    CritiqueOrderBlocked,
    UnpinnedModelBlocked,
    InsufficientPairsBlocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapMetrics {
    pub total_pairs: usize,
    pub consistent_win_a: usize,
    pub consistent_win_b: usize,
    pub consistent_ties: usize,
    pub primacy_flips: usize,
    pub recency_flips: usize,
    pub partial_flips: usize,
    pub total_flips: usize,
    pub swap_consistency_rate: f64,
    pub flip_rate: f64,
    pub position_bias_index: f64,
    pub swap_kappa: f64,
    pub strict_win_rate_a: f64,
    pub strict_win_rate_b: f64,
    pub tie_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseAdjudicationReport {
    pub case_id: String,
    pub stratum: String,
    pub severity: CaseSeverity,
    pub regime: PairwiseBiasRegime,
    pub passed: bool,
    pub swap_metrics: SwapMetrics,
    pub verbosity: VerbosityReport,
    pub failure_reasons: Vec<String>,
}

pub fn compute_cohen_kappa(
    pass1_choices: &[usize], // 0: A, 1: B, 2: Tie
    pass2_choices: &[usize], // 0: A, 1: B, 2: Tie
) -> f64 {
    let n = pass1_choices.len();
    if n == 0 {
        return 1.0;
    }

    let mut matrix = [[0_usize; 3]; 3];
    for (&p1, &p2) in pass1_choices.iter().zip(pass2_choices.iter()) {
        if p1 < 3 && p2 < 3 {
            matrix[p1][p2] += 1;
        }
    }

    let p_observed = (matrix[0][0] + matrix[1][1] + matrix[2][2]) as f64 / n as f64;

    let p1_marginals = [
        (matrix[0][0] + matrix[0][1] + matrix[0][2]) as f64 / n as f64,
        (matrix[1][0] + matrix[1][1] + matrix[1][2]) as f64 / n as f64,
        (matrix[2][0] + matrix[2][1] + matrix[2][2]) as f64 / n as f64,
    ];

    let p2_marginals = [
        (matrix[0][0] + matrix[1][0] + matrix[2][0]) as f64 / n as f64,
        (matrix[0][1] + matrix[1][1] + matrix[2][1]) as f64 / n as f64,
        (matrix[0][2] + matrix[1][2] + matrix[2][2]) as f64 / n as f64,
    ];

    let p_expected = p1_marginals[0] * p2_marginals[0]
        + p1_marginals[1] * p2_marginals[1]
        + p1_marginals[2] * p2_marginals[2];

    if (1.0 - p_expected).abs() < 1e-9 {
        1.0
    } else {
        ((p_observed - p_expected) / (1.0 - p_expected)).clamp(-1.0, 1.0)
    }
}

pub fn evaluate_case(
    case: &PairwiseCaseSpec,
    records: &[PairwiseComparisonRecord],
    bootstrap_iterations: usize,
    seed: u64,
) -> Result<CaseAdjudicationReport, PairError> {
    let mut failure_reasons = Vec::new();

    if records.len() < case.min_pairs {
        failure_reasons.push(format!(
            "insufficient pairs: got {} < min_pairs {}",
            records.len(),
            case.min_pairs
        ));
        return Ok(CaseAdjudicationReport {
            case_id: case.id.clone(),
            stratum: case.stratum.clone(),
            severity: case.severity,
            regime: PairwiseBiasRegime::InsufficientPairsBlocked,
            passed: false,
            swap_metrics: SwapMetrics {
                total_pairs: records.len(),
                consistent_win_a: 0,
                consistent_win_b: 0,
                consistent_ties: 0,
                primacy_flips: 0,
                recency_flips: 0,
                partial_flips: 0,
                total_flips: 0,
                swap_consistency_rate: 0.0,
                flip_rate: 1.0,
                position_bias_index: 0.0,
                swap_kappa: 0.0,
                strict_win_rate_a: 0.0,
                strict_win_rate_b: 0.0,
                tie_rate: 0.0,
            },
            verbosity: VerbosityReport {
                total_consistent_decisive: 0,
                longer_won_count: 0,
                shorter_won_count: 0,
                equal_length_count: 0,
                longer_win_rate: 0.5,
                length_correlation_r: 0.0,
                length_matched_pairs: 0,
                length_matched_win_rate_a: 0.5,
                raw_win_rate_a: 0.5,
                length_controlled_win_rate_a: 0.5,
                verbosity_inflation_delta: 0.0,
                lc_bootstrap_95_ci: (0.5, 0.5),
            },
            failure_reasons,
        });
    }

    let mut win_a = 0;
    let mut win_b = 0;
    let mut ties = 0;
    let mut primacy_flips = 0;
    let mut recency_flips = 0;
    let mut partial_flips = 0;

    let mut pass1_choices = Vec::with_capacity(records.len());
    let mut pass2_choices = Vec::with_capacity(records.len());
    let mut outcomes_and_lengths = Vec::with_capacity(records.len());

    for rec in records {
        let outcome = adjudicate_pair(
            rec,
            &case.candidate_a,
            &case.candidate_b,
            case.min_critique_words,
        )?;

        let p1_choice = match rec.forward_pass.verdict {
            crate::pair::SlotVerdict::Slot1 => 0, // Candidate A
            crate::pair::SlotVerdict::Slot2 => 1, // Candidate B
            crate::pair::SlotVerdict::Tie => 2,
        };

        let p2_choice = match rec.swapped_pass.verdict {
            crate::pair::SlotVerdict::Slot1 => 1, // Candidate B in Slot 1
            crate::pair::SlotVerdict::Slot2 => 0, // Candidate A in Slot 2
            crate::pair::SlotVerdict::Tie => 2,
        };

        pass1_choices.push(p1_choice);
        pass2_choices.push(p2_choice);

        match outcome {
            ConsistentOutcome::ConsistentWinA => win_a += 1,
            ConsistentOutcome::ConsistentWinB => win_b += 1,
            ConsistentOutcome::ConsistentTie => ties += 1,
            ConsistentOutcome::InconsistentPrimacyFlip => primacy_flips += 1,
            ConsistentOutcome::InconsistentRecencyFlip => recency_flips += 1,
            ConsistentOutcome::InconsistentPartialFlip => partial_flips += 1,
        }

        let a_stats = TextLengthStats::compute(&rec.candidate_a_response);
        let b_stats = TextLengthStats::compute(&rec.candidate_b_response);

        if case.length_policy.mode == crate::spec::LengthPolicyMode::RejectOverBudget {
            if a_stats.est_tokens > case.length_policy.max_tokens
                || b_stats.est_tokens > case.length_policy.max_tokens
            {
                failure_reasons.push(format!(
                    "pair '{}' exceeds token budget (max {}): A={}, B={}",
                    rec.pair_id,
                    case.length_policy.max_tokens,
                    a_stats.est_tokens,
                    b_stats.est_tokens
                ));
            }
        }

        outcomes_and_lengths.push((outcome, a_stats.est_tokens, b_stats.est_tokens));
    }

    let n = records.len();
    let total_flips = primacy_flips + recency_flips + partial_flips;
    let swap_consistency_rate = (win_a + win_b + ties) as f64 / n as f64;
    let flip_rate = total_flips as f64 / n as f64;
    let position_bias_index = (primacy_flips as f64 - recency_flips as f64) / n as f64;
    let swap_kappa = compute_cohen_kappa(&pass1_choices, &pass2_choices);

    let strict_win_rate_a = win_a as f64 / n as f64;
    let strict_win_rate_b = win_b as f64 / n as f64;
    let tie_rate = ties as f64 / n as f64;

    let swap_metrics = SwapMetrics {
        total_pairs: n,
        consistent_win_a: win_a,
        consistent_win_b: win_b,
        consistent_ties: ties,
        primacy_flips,
        recency_flips,
        partial_flips,
        total_flips,
        swap_consistency_rate,
        flip_rate,
        position_bias_index,
        swap_kappa,
        strict_win_rate_a,
        strict_win_rate_b,
        tie_rate,
    };

    let verbosity = analyze_verbosity(
        &outcomes_and_lengths,
        case.length_policy.length_matched_tau,
        bootstrap_iterations,
        seed,
    );

    if swap_consistency_rate < case.min_swap_consistency {
        failure_reasons.push(format!(
            "swap consistency {:.2}% < threshold {:.2}%",
            swap_consistency_rate * 100.0,
            case.min_swap_consistency * 100.0
        ));
    }

    if flip_rate > case.max_flip_rate {
        failure_reasons.push(format!(
            "flip rate {:.2}% > max allowable {:.2}%",
            flip_rate * 100.0,
            case.max_flip_rate * 100.0
        ));
    }

    if swap_kappa < case.min_swap_kappa {
        failure_reasons.push(format!(
            "swap agreement kappa {:.3} < threshold {:.3}",
            swap_kappa, case.min_swap_kappa
        ));
    }

    if verbosity.length_correlation_r.abs() > case.max_length_correlation {
        failure_reasons.push(format!(
            "verbosity correlation r={:.3} exceeds max allowable {:.3}",
            verbosity.length_correlation_r, case.max_length_correlation
        ));
    }

    if verbosity.verbosity_inflation_delta.abs() > case.max_verbosity_inflation {
        failure_reasons.push(format!(
            "verbosity inflation delta {:.3} exceeds threshold {:.3}",
            verbosity.verbosity_inflation_delta, case.max_verbosity_inflation
        ));
    }

    let regime = if !failure_reasons.is_empty() {
        if failure_reasons.iter().any(|r| r.contains("token budget")) {
            PairwiseBiasRegime::LengthBudgetExceeded
        } else if swap_consistency_rate < case.min_swap_consistency
            || flip_rate > case.max_flip_rate
        {
            if primacy_flips >= recency_flips {
                PairwiseBiasRegime::PrimacyBiasSkewed
            } else {
                PairwiseBiasRegime::RecencyBiasSkewed
            }
        } else if verbosity.length_correlation_r.abs() > case.max_length_correlation
            || verbosity.verbosity_inflation_delta.abs() > case.max_verbosity_inflation
        {
            PairwiseBiasRegime::VerbositySkewDistorted
        } else {
            PairwiseBiasRegime::PrimacyBiasSkewed
        }
    } else {
        PairwiseBiasRegime::EquilibriumPoised
    };

    let passed = failure_reasons.is_empty();

    Ok(CaseAdjudicationReport {
        case_id: case.id.clone(),
        stratum: case.stratum.clone(),
        severity: case.severity,
        regime,
        passed,
        swap_metrics,
        verbosity,
        failure_reasons,
    })
}
