use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseEconomicsReport {
    pub total_pairs: usize,
    pub single_pass_tokens: usize,
    pub double_pass_tokens: usize,
    pub additional_pass_cost_usd: f64,
    pub prevented_false_winner_cost_usd: f64,
    pub length_capping_tokens_saved: usize,
    pub length_capping_savings_usd: f64,
    pub net_roi_multiplier: f64,
}

pub fn calculate_pairwise_economics(
    total_pairs: usize,
    avg_tokens_per_response: usize,
    avg_tokens_post_capping: usize,
    quarantined_flips: usize,
    token_cost_per_million: f64,
    false_ship_cost_usd: f64,
) -> PairwiseEconomicsReport {
    let single_pass_tokens = total_pairs * (avg_tokens_per_response * 2 + 150);
    let double_pass_tokens = total_pairs * 2 * (avg_tokens_post_capping * 2 + 150);

    let token_cost_rate = token_cost_per_million / 1_000_000.0;
    let additional_pass_cost_usd =
        (double_pass_tokens as f64 - single_pass_tokens as f64).max(0.0) * token_cost_rate;

    let tokens_saved =
        total_pairs * 2 * (avg_tokens_per_response.saturating_sub(avg_tokens_post_capping) * 2);
    let length_capping_savings_usd = tokens_saved as f64 * token_cost_rate;

    let prevented_false_winner_cost_usd = if quarantined_flips > 0 {
        false_ship_cost_usd
    } else {
        0.0
    };

    let total_benefit = prevented_false_winner_cost_usd + length_capping_savings_usd;
    let total_cost = additional_pass_cost_usd.max(0.01);
    let net_roi_multiplier = total_benefit / total_cost;

    PairwiseEconomicsReport {
        total_pairs,
        single_pass_tokens,
        double_pass_tokens,
        additional_pass_cost_usd,
        prevented_false_winner_cost_usd,
        length_capping_tokens_saved: tokens_saved,
        length_capping_savings_usd,
        net_roi_multiplier,
    }
}
