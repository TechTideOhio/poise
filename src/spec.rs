use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpecError {
    #[error("failed to read suite specification from {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse TOML suite specification: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("strata weights must sum to 1.00 (got {0:.4})")]
    InvalidWeights(f64),
    #[error("case '{0}' has invalid model identifier '{1}': must pin dated snapshot or immutable digest")]
    UnpinnedModel(String, String),
    #[error("case '{0}' has non-zero judge temperature {1:.2}: temperature must be pinned to 0.0")]
    NonZeroTemperature(String, f64),
    #[error("case '{0}' min_pairs must be at least 10 (got {1})")]
    InsufficientMinPairs(String, usize),
    #[error("case '{0}' min_swap_consistency must be in range (0.0, 1.0] (got {1:.2})")]
    InvalidSwapConsistency(String, f64),
    #[error("case '{0}' length cap {1} must be positive")]
    InvalidLengthCap(String, usize),
    #[error("case '{0}' not found in suite")]
    CaseNotFound(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaseSeverity {
    Blocker,
    Graded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LengthPolicyMode {
    CapAtBudget,
    RejectOverBudget,
    NormalizeRatio,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthPolicySpec {
    pub mode: LengthPolicyMode,
    pub max_tokens: usize,
    pub max_chars: usize,
    #[serde(default)]
    pub strip_fluff: bool,
    #[serde(default = "default_tau_matched")]
    pub length_matched_tau: f64,
}

fn default_tau_matched() -> f64 {
    0.10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseRubricSpec {
    pub title: String,
    pub instruction: String,
    #[serde(default)]
    pub required_tokens: Vec<String>,
    #[serde(default)]
    pub forbidden_tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseCaseSpec {
    pub id: String,
    pub stratum: String,
    pub severity: CaseSeverity,
    pub description: String,
    pub pairs_path: String,
    pub candidate_a: String,
    pub candidate_b: String,
    pub pinned_model: String,
    pub judge_temperature: f64,
    pub prompt_version: String,
    pub min_pairs: usize,
    pub min_swap_consistency: f64,
    pub max_flip_rate: f64,
    pub min_swap_kappa: f64,
    pub max_length_correlation: f64,
    pub max_verbosity_inflation: f64,
    pub min_critique_words: usize,
    pub length_policy: LengthPolicySpec,
    pub rubric: PairwiseRubricSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrataWeights {
    pub critical: f64,
    pub core: f64,
    pub hygiene: f64,
}

impl StrataWeights {
    pub fn validate(&self) -> Result<(), SpecError> {
        let sum = self.critical + self.core + self.hygiene;
        if (sum - 1.0).abs() > 0.001 {
            return Err(SpecError::InvalidWeights(sum));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteMeta {
    pub name: String,
    pub version: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteSpec {
    pub suite: SuiteMeta,
    pub weights: StrataWeights,
    pub cases: Vec<PairwiseCaseSpec>,
}

impl SuiteSpec {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, SpecError> {
        let p = path.as_ref();
        let content = std::fs::read_to_string(p).map_err(|e| SpecError::Io {
            path: p.display().to_string(),
            source: e,
        })?;
        let spec: Self = toml::from_str(&content)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        self.weights.validate()?;

        let pinned_model_regex = regex::Regex::new(
            r"^([a-z0-9._-]+-(20\d{2}-\d{2}-\d{2}|20\d{6})|[a-z0-9._-]+@sha256:[a-f0-9]{64})$",
        )
        .map_err(|_| SpecError::InvalidWeights(0.0))?;

        for case in &self.cases {
            if !pinned_model_regex.is_match(&case.pinned_model) {
                return Err(SpecError::UnpinnedModel(
                    case.id.clone(),
                    case.pinned_model.clone(),
                ));
            }

            if case.judge_temperature.abs() > 1e-6 {
                return Err(SpecError::NonZeroTemperature(
                    case.id.clone(),
                    case.judge_temperature,
                ));
            }

            if case.min_pairs < 10 {
                return Err(SpecError::InsufficientMinPairs(
                    case.id.clone(),
                    case.min_pairs,
                ));
            }

            if case.min_swap_consistency <= 0.0 || case.min_swap_consistency > 1.0 {
                return Err(SpecError::InvalidSwapConsistency(
                    case.id.clone(),
                    case.min_swap_consistency,
                ));
            }

            if case.length_policy.max_tokens == 0 || case.length_policy.max_chars == 0 {
                return Err(SpecError::InvalidLengthCap(
                    case.id.clone(),
                    case.length_policy.max_tokens,
                ));
            }
        }

        Ok(())
    }

    pub fn get_case(&self, id: &str) -> Result<&PairwiseCaseSpec, SpecError> {
        self.cases
            .iter()
            .find(|c| c.id == id)
            .ok_or_else(|| SpecError::CaseNotFound(id.to_string()))
    }

    pub fn strata_cases(&self) -> BTreeMap<String, Vec<&PairwiseCaseSpec>> {
        let mut map: BTreeMap<String, Vec<&PairwiseCaseSpec>> = BTreeMap::new();
        for case in &self.cases {
            map.entry(case.stratum.clone()).or_default().push(case);
        }
        map
    }
}
