#![forbid(unsafe_code)]

pub mod adjudicator;
pub mod economics;
pub mod envelope;
pub mod ingest;
pub mod length;
pub mod lockfile;
pub mod pair;
pub mod rating;
pub mod repo_audit;
pub mod report;
pub mod simulator;
pub mod spec;
pub mod verbosity;

pub use adjudicator::{
    compute_cohen_kappa, evaluate_case, CaseAdjudicationReport, PairwiseBiasRegime, SwapMetrics,
};
pub use economics::{calculate_pairwise_economics, PairwiseEconomicsReport};
pub use envelope::{EnvelopeStatus, HarnessEnvelope};
pub use length::{
    cap_text_at_budget, normalize_and_cap_pair, strip_fluff_boilerplate, NormalizedPairStats,
    TextLengthStats,
};
pub use lockfile::{generate_lockfile, verify_lockfile, PoiseLockfile};
pub use pair::{
    adjudicate_pair, compute_pair_hash, load_pairs, validate_critique_before_verdict,
    ConsistentOutcome, PairError, PairwiseComparisonRecord, PassJudgment, SlotVerdict, SplitMix64,
};
pub use rating::{
    rank_tournament, solve_bradley_terry, CandidateTournamentRating, TournamentReport,
};
pub use repo_audit::{audit_repository, RepoAuditReport};
pub use report::{export_mermaid_flowchart, export_otel_spans, render_reference_table};
pub use simulator::{simulate_pair_judgments, JudgeProfile};
pub use spec::{
    CaseSeverity, LengthPolicyMode, LengthPolicySpec, PairwiseCaseSpec, PairwiseRubricSpec,
    SpecError, StrataWeights, SuiteMeta, SuiteSpec,
};
pub use verbosity::{analyze_verbosity, fit_logistic_lc_win_rate, VerbosityReport};
