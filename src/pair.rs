use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PairError {
    #[error("failed to read pairs file {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("line {line}: failed to parse JSONL record: {source}")]
    Parse {
        line: usize,
        source: serde_json::Error,
    },
    #[error("pair '{pair_id}': forward pass must map slot 1 to candidate A ('{expected_a}') and slot 2 to candidate B ('{expected_b}')")]
    InvalidForwardMapping {
        pair_id: String,
        expected_a: String,
        expected_b: String,
    },
    #[error("pair '{pair_id}': swapped pass must invert candidates: slot 1 must be candidate B ('{expected_b}') and slot 2 must be candidate A ('{expected_a}')")]
    InvalidSwappedMapping {
        pair_id: String,
        expected_a: String,
        expected_b: String,
    },
    #[error("pair '{pair_id}' pass '{pass}': raw response emits 'verdict' at byte offset {verdict_pos} before 'critique' at byte offset {critique_pos}")]
    CritiqueOrderViolation {
        pair_id: String,
        pass: String,
        critique_pos: usize,
        verdict_pos: usize,
    },
    #[error("pair '{pair_id}' pass '{pass}': critique is too short ({word_count} words < required {min_words})")]
    CritiqueTooShort {
        pair_id: String,
        pass: String,
        word_count: usize,
        min_words: usize,
    },
    #[error("provenance chain broken at pair '{pair_id}': expected prev_hash '{expected}' but found '{found}'")]
    BrokenProvenance {
        pair_id: String,
        expected: String,
        found: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SlotVerdict {
    Slot1,
    Slot2,
    Tie,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsistentOutcome {
    ConsistentWinA,
    ConsistentWinB,
    ConsistentTie,
    InconsistentPrimacyFlip,
    InconsistentRecencyFlip,
    InconsistentPartialFlip,
}

impl ConsistentOutcome {
    pub fn is_consistent_decisive(&self) -> bool {
        matches!(self, Self::ConsistentWinA | Self::ConsistentWinB)
    }

    pub fn is_consistent(&self) -> bool {
        matches!(
            self,
            Self::ConsistentWinA | Self::ConsistentWinB | Self::ConsistentTie
        )
    }

    pub fn is_flip(&self) -> bool {
        !self.is_consistent()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassJudgment {
    pub slot1_candidate: String,
    pub slot2_candidate: String,
    pub critique: String,
    pub verdict: SlotVerdict,
    pub raw_response: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparisonRecord {
    pub pair_id: String,
    pub prompt: String,
    pub candidate_a_response: String,
    pub candidate_b_response: String,
    pub forward_pass: PassJudgment,
    pub swapped_pass: PassJudgment,
    #[serde(default)]
    pub prev_hash: String,
    #[serde(default)]
    pub pair_hash: String,
}

pub fn validate_critique_before_verdict(
    pair_id: &str,
    pass_name: &str,
    judgment: &PassJudgment,
    min_critique_words: usize,
) -> Result<(), PairError> {
    let word_count = judgment.critique.split_whitespace().count();
    if word_count < min_critique_words {
        return Err(PairError::CritiqueTooShort {
            pair_id: pair_id.to_string(),
            pass: pass_name.to_string(),
            word_count,
            min_words: min_critique_words,
        });
    }

    let raw = &judgment.raw_response;
    let critique_idx = raw
        .find("\"critique\"")
        .or_else(|| raw.find("'critique'"))
        .unwrap_or(0);
    let verdict_idx = raw
        .find("\"verdict\"")
        .or_else(|| raw.find("'verdict'"))
        .or_else(|| raw.find("\"winner\""))
        .or_else(|| raw.find("'winner'"));

    if let Some(v_idx) = verdict_idx {
        if v_idx < critique_idx {
            return Err(PairError::CritiqueOrderViolation {
                pair_id: pair_id.to_string(),
                pass: pass_name.to_string(),
                critique_pos: critique_idx,
                verdict_pos: v_idx,
            });
        }
    }

    Ok(())
}

pub fn adjudicate_pair(
    record: &PairwiseComparisonRecord,
    candidate_a: &str,
    candidate_b: &str,
    min_critique_words: usize,
) -> Result<ConsistentOutcome, PairError> {
    if record.forward_pass.slot1_candidate != candidate_a
        || record.forward_pass.slot2_candidate != candidate_b
    {
        return Err(PairError::InvalidForwardMapping {
            pair_id: record.pair_id.clone(),
            expected_a: candidate_a.to_string(),
            expected_b: candidate_b.to_string(),
        });
    }

    if record.swapped_pass.slot1_candidate != candidate_b
        || record.swapped_pass.slot2_candidate != candidate_a
    {
        return Err(PairError::InvalidSwappedMapping {
            pair_id: record.pair_id.clone(),
            expected_a: candidate_a.to_string(),
            expected_b: candidate_b.to_string(),
        });
    }

    validate_critique_before_verdict(
        &record.pair_id,
        "forward",
        &record.forward_pass,
        min_critique_words,
    )?;
    validate_critique_before_verdict(
        &record.pair_id,
        "swapped",
        &record.swapped_pass,
        min_critique_words,
    )?;

    let f_verdict = record.forward_pass.verdict;
    let s_verdict = record.swapped_pass.verdict;

    let outcome = match (f_verdict, s_verdict) {
        (SlotVerdict::Slot1, SlotVerdict::Slot2) => ConsistentOutcome::ConsistentWinA,
        (SlotVerdict::Slot2, SlotVerdict::Slot1) => ConsistentOutcome::ConsistentWinB,
        (SlotVerdict::Tie, SlotVerdict::Tie) => ConsistentOutcome::ConsistentTie,
        (SlotVerdict::Slot1, SlotVerdict::Slot1) => ConsistentOutcome::InconsistentPrimacyFlip,
        (SlotVerdict::Slot2, SlotVerdict::Slot2) => ConsistentOutcome::InconsistentRecencyFlip,
        _ => ConsistentOutcome::InconsistentPartialFlip,
    };

    Ok(outcome)
}

pub fn compute_pair_hash(
    record: &PairwiseComparisonRecord,
    prev_hash: &str,
) -> Result<String, PairError> {
    let mut hasher = Sha256::new();
    hasher.update(b"tideos-poise-pair-v1\0");
    hasher.update(prev_hash.as_bytes());
    hasher.update(record.pair_id.as_bytes());
    hasher.update(record.prompt.as_bytes());
    hasher.update(record.candidate_a_response.as_bytes());
    hasher.update(record.candidate_b_response.as_bytes());
    hasher.update(record.forward_pass.raw_response.as_bytes());
    hasher.update(record.swapped_pass.raw_response.as_bytes());
    let digest = hasher.finalize();
    Ok(format!("{digest:x}"))
}

pub fn load_pairs(path: impl AsRef<Path>) -> Result<Vec<PairwiseComparisonRecord>, PairError> {
    let p = path.as_ref();
    let file = File::open(p).map_err(|e| PairError::Io {
        path: p.display().to_string(),
        source: e,
    })?;
    let reader = BufReader::new(file);

    let mut records = Vec::new();
    let mut prev_hash = String::from("genesis");

    for (idx, line) in reader.lines().enumerate() {
        let line_content = line.map_err(|e| PairError::Io {
            path: p.display().to_string(),
            source: e,
        })?;
        let trimmed = line_content.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut record: PairwiseComparisonRecord =
            serde_json::from_str(trimmed).map_err(|e| PairError::Parse {
                line: idx + 1,
                source: e,
            })?;

        if !record.prev_hash.is_empty() && record.prev_hash != prev_hash {
            return Err(PairError::BrokenProvenance {
                pair_id: record.pair_id.clone(),
                expected: prev_hash,
                found: record.prev_hash.clone(),
            });
        }

        let expected_hash = compute_pair_hash(&record, &prev_hash)?;
        if !record.pair_hash.is_empty() && record.pair_hash != expected_hash {
            return Err(PairError::BrokenProvenance {
                pair_id: record.pair_id.clone(),
                expected: expected_hash,
                found: record.pair_hash.clone(),
            });
        }

        record.prev_hash = prev_hash;
        record.pair_hash = expected_hash.clone();
        prev_hash = expected_hash;

        records.push(record);
    }

    Ok(records)
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}
