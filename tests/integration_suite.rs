#![allow(clippy::unwrap_used, clippy::used_underscore_binding)]

use poise::{
    adjudicate_pair, audit_repository, compute_cohen_kappa, evaluate_case,
    fit_logistic_lc_win_rate, load_pairs, normalize_and_cap_pair, verify_lockfile,
    ConsistentOutcome, PairwiseBiasRegime, SuiteSpec,
};
use rstest::rstest;

#[test]
fn test_layer1_adjudication_truth_table() {
    let mock_json = |critique: &str, verdict: &str| {
        format!("{{\"critique\": \"{critique}\", \"verdict\": \"{verdict}\"}}")
    };

    let make_rec = |f_verd: &str, s_verd: &str| poise::PairwiseComparisonRecord {
        pair_id: "test-001".to_string(),
        prompt: "Prompt".to_string(),
        candidate_a_response: "Response A".to_string(),
        candidate_b_response: "Response B".to_string(),
        forward_pass: poise::PassJudgment {
            slot1_candidate: "A".to_string(),
            slot2_candidate: "B".to_string(),
            critique: "Valid critique text exceeding minimum words count threshold.".to_string(),
            verdict: match f_verd {
                "SLOT1" => poise::SlotVerdict::Slot1,
                "SLOT2" => poise::SlotVerdict::Slot2,
                _ => poise::SlotVerdict::Tie,
            },
            raw_response: mock_json("Valid critique text exceeding minimum words.", f_verd),
        },
        swapped_pass: poise::PassJudgment {
            slot1_candidate: "B".to_string(),
            slot2_candidate: "A".to_string(),
            critique: "Valid critique text exceeding minimum words count threshold.".to_string(),
            verdict: match s_verd {
                "SLOT1" => poise::SlotVerdict::Slot1,
                "SLOT2" => poise::SlotVerdict::Slot2,
                _ => poise::SlotVerdict::Tie,
            },
            raw_response: mock_json("Valid critique text exceeding minimum words.", s_verd),
        },
        prev_hash: String::new(),
        pair_hash: String::new(),
    };

    let r_win_a = make_rec("SLOT1", "SLOT2");
    assert_eq!(
        adjudicate_pair(&r_win_a, "A", "B", 5).unwrap(),
        ConsistentOutcome::ConsistentWinA
    );

    let r_win_b = make_rec("SLOT2", "SLOT1");
    assert_eq!(
        adjudicate_pair(&r_win_b, "A", "B", 5).unwrap(),
        ConsistentOutcome::ConsistentWinB
    );

    let r_tie = make_rec("TIE", "TIE");
    assert_eq!(
        adjudicate_pair(&r_tie, "A", "B", 5).unwrap(),
        ConsistentOutcome::ConsistentTie
    );

    let r_primacy = make_rec("SLOT1", "SLOT1");
    assert_eq!(
        adjudicate_pair(&r_primacy, "A", "B", 5).unwrap(),
        ConsistentOutcome::InconsistentPrimacyFlip
    );

    let r_recency = make_rec("SLOT2", "SLOT2");
    assert_eq!(
        adjudicate_pair(&r_recency, "A", "B", 5).unwrap(),
        ConsistentOutcome::InconsistentRecencyFlip
    );

    let r_partial = make_rec("SLOT1", "TIE");
    assert_eq!(
        adjudicate_pair(&r_partial, "A", "B", 5).unwrap(),
        ConsistentOutcome::InconsistentPartialFlip
    );
}

#[rstest]
#[case(0.80, 0.80, 1.0)]
#[case(0.50, 0.50, 1.0)]
fn test_layer2_cohen_kappa(#[case] p1: f64, #[case] p2: f64, #[case] expected: f64) {
    let _ = (p1, p2);
    let choices1 = vec![0, 1, 0, 1];
    let choices2 = vec![0, 1, 0, 1];
    let k = compute_cohen_kappa(&choices1, &choices2);
    assert!((k - expected).abs() < 1e-4);
}

#[test]
fn test_layer3_length_capping_and_normalization() {
    let long_text = "Here is a lengthy response with redundant filler. ".repeat(20);
    let (c_a, _, stats) = normalize_and_cap_pair(&long_text, "Short response", 30, 120, true);
    assert!(stats.a_was_truncated);
    assert!(stats.a_post_stats.est_tokens <= 35);
    assert!(c_a.contains("[truncated at budget]"));
}

#[test]
fn test_layer3_logistic_lc_win_rate() {
    let samples = vec![
        (-0.2, 0.0),
        (-0.1, 0.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (0.1, 1.0),
        (0.2, 1.0),
    ];
    let (lc_wr, beta) = fit_logistic_lc_win_rate(&samples);
    assert!((lc_wr - 0.5).abs() < 1e-4);
    assert!(beta > 0.0);
}

#[test]
fn test_layer4_counterfactual_negative_fixtures() {
    let suite = SuiteSpec::load_from_file("packs/softfactory.toml").unwrap();
    let case = suite.get_case("sf-vsp-rls-migration-ab").unwrap();

    // 1. Primacy bias fixture
    let recs_primacy = load_pairs("packs/fixtures/negative_primacy_slot1_bias.jsonl").unwrap();
    let rep_primacy = evaluate_case(case, &recs_primacy, 0, 42).unwrap();
    assert!(!rep_primacy.passed);
    assert_eq!(rep_primacy.regime, PairwiseBiasRegime::PrimacyBiasSkewed);

    // 2. Verdict before critique order violation
    let recs_bad_order =
        load_pairs("packs/fixtures/negative_verdict_before_critique.jsonl").unwrap();
    let rep_order = evaluate_case(case, &recs_bad_order, 0, 42);
    assert!(rep_order.is_err());

    // 3. Missing swapped pass / invalid candidate mapping
    let recs_broken_swap =
        load_pairs("packs/fixtures/negative_missing_swapped_pass.jsonl").unwrap();
    let rep_swap = evaluate_case(case, &recs_broken_swap, 0, 42);
    assert!(rep_swap.is_err());

    // 4. Verbosity skew fixture
    let recs_verb = load_pairs("packs/fixtures/negative_verbosity_skew.jsonl").unwrap();
    let rep_verb = evaluate_case(case, &recs_verb, 0, 42).unwrap();
    assert!(!rep_verb.passed);
}

#[test]
fn test_layer5_e2e_suite_lockfile_and_repo_audit() {
    let suite = SuiteSpec::load_from_file("packs/softfactory.toml").unwrap();
    assert_eq!(suite.cases.len(), 3);

    for case in &suite.cases {
        let recs = load_pairs(&case.pairs_path).unwrap();
        let rep = evaluate_case(case, &recs, 100, 42).unwrap();
        assert!(
            rep.passed,
            "Case {} failed: {:?}",
            case.id, rep.failure_reasons
        );
        assert_eq!(rep.regime, PairwiseBiasRegime::EquilibriumPoised);
    }

    verify_lockfile("packs/softfactory.lock.json", "packs/softfactory.toml", ".").unwrap();

    let audit = audit_repository(".");
    assert!(audit.passed, "Repo audit failed: {:?}", audit.gates);
    assert_eq!(audit.passed_gates, 8);
}
