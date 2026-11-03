use crate::pair::{PassJudgment, SlotVerdict, SplitMix64};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum JudgeProfile {
    PoisedDoubleBlind,
    PrimacySlot1Biased,
    RecencySlot2Biased,
    VerbosityChaser,
    SinglePassCheater,
}

pub fn simulate_pair_judgments(
    pair_id: &str,
    candidate_a: &str,
    candidate_b: &str,
    prompt: &str,
    response_a: &str,
    response_b: &str,
    profile: JudgeProfile,
    prng: &mut SplitMix64,
) -> (PassJudgment, PassJudgment) {
    let len_a = response_a.split_whitespace().count();
    let len_b = response_b.split_whitespace().count();

    let (f_verdict, s_verdict, f_critique, s_critique) = match profile {
        JudgeProfile::PoisedDoubleBlind => {
            let roll = prng.next_f64();
            if roll < 0.48 {
                (
                    SlotVerdict::Slot1,
                    SlotVerdict::Slot2,
                    "Critique: Candidate A provides superior boundary guarantees and proper tenant verification."
                        .to_string(),
                    "Critique: Candidate A in slot 2 provides superior boundary guarantees."
                        .to_string(),
                )
            } else if roll < 0.92 {
                (
                    SlotVerdict::Slot2,
                    SlotVerdict::Slot1,
                    "Critique: Candidate B provides more robust SQL safety checks and cleaner migrations."
                        .to_string(),
                    "Critique: Candidate B in slot 1 provides more robust SQL safety checks."
                        .to_string(),
                )
            } else {
                (
                    SlotVerdict::Tie,
                    SlotVerdict::Tie,
                    "Critique: Both candidates present equivalent correctness and security trade-offs."
                        .to_string(),
                    "Critique: Both candidates remain equivalent in swapped presentation."
                        .to_string(),
                )
            }
        }
        JudgeProfile::PrimacySlot1Biased => {
            let roll = prng.next_f64();
            if roll < 0.85 {
                (
                    SlotVerdict::Slot1,
                    SlotVerdict::Slot1,
                    "Critique: Slot 1 is immediately well-structured and concise.".to_string(),
                    "Critique: Slot 1 in swapped pass is also immediately compelling.".to_string(),
                )
            } else {
                (
                    SlotVerdict::Slot2,
                    SlotVerdict::Slot2,
                    "Critique: Slot 2 details are acceptable.".to_string(),
                    "Critique: Slot 2 details are acceptable.".to_string(),
                )
            }
        }
        JudgeProfile::RecencySlot2Biased => {
            let roll = prng.next_f64();
            if roll < 0.85 {
                (
                    SlotVerdict::Slot2,
                    SlotVerdict::Slot2,
                    "Critique: Slot 2 left a stronger final impression on review.".to_string(),
                    "Critique: Slot 2 in swapped pass left a stronger final impression."
                        .to_string(),
                )
            } else {
                (
                    SlotVerdict::Slot1,
                    SlotVerdict::Slot1,
                    "Critique: Slot 1 is chosen.".to_string(),
                    "Critique: Slot 1 is chosen.".to_string(),
                )
            }
        }
        JudgeProfile::VerbosityChaser => {
            if len_a > len_b {
                (
                    SlotVerdict::Slot1,
                    SlotVerdict::Slot2,
                    "Critique: Response A is substantially longer and provides detailed explanations."
                        .to_string(),
                    "Critique: Response A in slot 2 is more detailed and thorough.".to_string(),
                )
            } else {
                (
                    SlotVerdict::Slot2,
                    SlotVerdict::Slot1,
                    "Critique: Response B has more paragraphs and extensive checklists."
                        .to_string(),
                    "Critique: Response B in slot 1 has more paragraphs and extensive checklists."
                        .to_string(),
                )
            }
        }
        JudgeProfile::SinglePassCheater => (
            SlotVerdict::Slot1,
            SlotVerdict::Slot1,
            "Critique: Forward pass only.".to_string(),
            "Critique: Swapped pass skipped.".to_string(),
        ),
    };

    let f_raw = format!(
        "{{\"critique\": \"{}\", \"verdict\": \"{:?}\"}}",
        f_critique, f_verdict
    );
    let s_raw = format!(
        "{{\"critique\": \"{}\", \"verdict\": \"{:?}\"}}",
        s_critique, s_verdict
    );

    let f_judgment = PassJudgment {
        slot1_candidate: candidate_a.to_string(),
        slot2_candidate: candidate_b.to_string(),
        critique: f_critique,
        verdict: f_verdict,
        raw_response: f_raw,
    };

    let s_judgment = PassJudgment {
        slot1_candidate: candidate_b.to_string(),
        slot2_candidate: candidate_a.to_string(),
        critique: s_critique,
        verdict: s_verdict,
        raw_response: s_raw,
    };

    let _ = (pair_id, prompt);

    (f_judgment, s_judgment)
}
