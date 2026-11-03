#![forbid(unsafe_code)]

use clap::{Args, Parser, Subcommand};
use poise::{
    audit_repository, evaluate_case, export_mermaid_flowchart, export_otel_spans,
    generate_lockfile, load_pairs, normalize_and_cap_pair, rank_tournament, render_reference_table,
    simulate_pair_judgments, verify_lockfile, EnvelopeStatus, HarnessEnvelope, JudgeProfile,
    PairwiseComparisonRecord, SplitMix64, SuiteSpec,
};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "poise",
    author = "Alex Cinovoj <alex@techtideai.io>",
    version = "0.1.0",
    about = "Control LLM judge position bias and verbosity skew in pairwise A/B comparisons",
    long_about = "Poise provides double-blind swapped order evaluation, consistent verdict adjudication, deterministic length capping, and length-controlled win rate estimation for TideOS Eval Kit."
)]
struct Cli {
    #[arg(long, global = true, help = "Emit machine-readable JSON envelope")]
    json: bool,

    #[arg(
        long,
        global = true,
        default_value = "true",
        help = "Ring terminal audio alert bell on failure"
    )]
    bell: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(
        about = "Display reference position bias and verbosity skew distortion physics table"
    )]
    Table(TableArgs),

    #[command(about = "Initialize default softfactory pairwise suite configuration")]
    Init(InitArgs),

    #[command(about = "Normalize, clean, and cap candidate response lengths in a pairs file")]
    Normalize(NormalizeArgs),

    #[command(about = "Score pairwise double-pass comparisons against a suite case")]
    Score(ScoreArgs),

    #[command(about = "Run all suite cases and verify gating thresholds")]
    Check(CheckArgs),

    #[command(about = "Compare two pairwise evaluation datasets for bias regressions")]
    Compare(CompareArgs),

    #[command(
        about = "Rank candidate agents/models via Bradley-Terry and Elo on consistent pairs"
    )]
    Rank(RankArgs),

    #[command(about = "Simulate pairwise judge evaluations under synthetic bias profiles")]
    Simulate(SimulateArgs),

    #[command(about = "Generate sha256-framed-v1 cryptographic lockfile")]
    Lock(LockArgs),

    #[command(about = "Verify suite datasets against cryptographic lockfile")]
    Verify(VerifyArgs),

    #[command(about = "Export suite reports to Mermaid, OpenTelemetry, or reval formats")]
    Export(ExportArgs),

    #[command(about = "Ingest pairs from agent harness runs")]
    Ingest(IngestArgs),

    #[command(about = "Audit repository for Soft Factory OSS hygiene and zero em-dashes")]
    Repo(RepoArgs),
}

#[derive(Args)]
struct TableArgs {
    #[arg(long, default_value = "0.50")]
    true_win_rate: f64,
    #[arg(long, default_value = "0.20")]
    primacy_bias: f64,
    #[arg(long, default_value = "0.15")]
    verbosity_skew: f64,
}

#[derive(Args)]
struct InitArgs {
    #[arg(long, default_value = "softfactory")]
    pack: String,
    #[arg(long, default_value = "packs/softfactory.toml")]
    out: PathBuf,
}

#[derive(Args)]
struct NormalizeArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "256")]
    max_tokens: usize,
    #[arg(long, default_value = "1024")]
    max_chars: usize,
    #[arg(long, default_value = "true")]
    strip_fluff: bool,
}

#[derive(Args)]
struct ScoreArgs {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    case: String,
    #[arg(long)]
    pairs: Option<PathBuf>,
    #[arg(long, default_value = "2000")]
    bootstrap_iterations: usize,
    #[arg(long, default_value = "42")]
    seed: u64,
}

#[derive(Args)]
struct CheckArgs {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long)]
    lock: Option<PathBuf>,
    #[arg(long, default_value = "2000")]
    bootstrap_iterations: usize,
    #[arg(long, default_value = "42")]
    seed: u64,
}

#[derive(Args)]
struct CompareArgs {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    case: String,
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
}

#[derive(Args)]
struct RankArgs {
    #[arg(long)]
    pairs: PathBuf,
    #[arg(long, default_value = "2000")]
    bootstrap_iterations: usize,
    #[arg(long, default_value = "42")]
    seed: u64,
}

#[derive(Args)]
struct SimulateArgs {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    case: String,
    #[arg(long)]
    profile: JudgeProfile,
    #[arg(long, default_value = "42")]
    seed: u64,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct LockArgs {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct VerifyArgs {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long)]
    lock: PathBuf,
}

#[derive(Args)]
struct ExportArgs {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long, default_value = "mermaid")]
    format: String,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct IngestArgs {
    #[arg(long, default_value = "Tools/agent-harness/runs")]
    runs_dir: PathBuf,
    #[arg(long, default_value = "claude")]
    candidate_a: String,
    #[arg(long, default_value = "codex")]
    candidate_b: String,
}

#[derive(Args)]
struct RepoArgs {
    #[arg(default_value = ".")]
    path: PathBuf,
}

fn ring_terminal_bell() {
    eprint!("\x07");
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match execute_subcommand(cli.command, cli.json, cli.bell) {
        Ok(code) => code,
        Err(err) => {
            if cli.bell {
                ring_terminal_bell();
            }
            if cli.json {
                let env = HarnessEnvelope::<()>::error(
                    format!("{err:#}"),
                    vec!["inspect error logs or rerun with --help".to_string()],
                    vec![],
                    (),
                );
                println!("{}", serde_json::to_string_pretty(&env).unwrap_or_default());
            } else {
                eprintln!("Error: {err:#}");
            }
            ExitCode::from(2)
        }
    }
}

fn execute_subcommand(cmd: Commands, is_json: bool, bell: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        Commands::Table(args) => {
            let table =
                render_reference_table(args.true_win_rate, args.primacy_bias, args.verbosity_skew);
            if is_json {
                let env = HarnessEnvelope::success(
                    "Position bias and verbosity skew distortion physics table rendered",
                    vec![],
                    serde_json::json!({ "table": table }),
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                print!("{table}");
            }
            Ok(ExitCode::SUCCESS)
        }

        Commands::Init(args) => {
            if let Some(parent) = args.out.parent() {
                fs::create_dir_all(parent)?;
            }
            let default_toml = include_str!("../packs/softfactory.toml");
            fs::write(&args.out, default_toml)?;
            if is_json {
                let env = HarnessEnvelope::success(
                    format!("Initialized suite specification at {}", args.out.display()),
                    vec![args.out.display().to_string()],
                    (),
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!(
                    "Initialized softfactory suite specification at {}",
                    args.out.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }

        Commands::Normalize(args) => {
            let file = File::open(&args.input)?;
            let reader = BufReader::new(file);

            if let Some(parent) = args.out.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out_file = File::create(&args.out)?;
            let mut count = 0;
            let mut trunc_a = 0;
            let mut trunc_b = 0;

            for line in reader.lines() {
                let text = line?;
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let mut rec: PairwiseComparisonRecord = serde_json::from_str(trimmed)?;
                let (c_a, c_b, stats) = normalize_and_cap_pair(
                    &rec.candidate_a_response,
                    &rec.candidate_b_response,
                    args.max_tokens,
                    args.max_chars,
                    args.strip_fluff,
                );
                rec.candidate_a_response = c_a;
                rec.candidate_b_response = c_b;
                if stats.a_was_truncated {
                    trunc_a += 1;
                }
                if stats.b_was_truncated {
                    trunc_b += 1;
                }
                writeln!(out_file, "{}", serde_json::to_string(&rec)?)?;
                count += 1;
            }

            if is_json {
                let env = HarnessEnvelope::success(
                    format!(
                        "Normalized {count} pairs: truncated {trunc_a} candidate A responses, {trunc_b} candidate B responses"
                    ),
                    vec![args.out.display().to_string()],
                    serde_json::json!({
                        "pairs_count": count,
                        "truncated_a": trunc_a,
                        "truncated_b": trunc_b,
                    }),
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!(
                    "Normalized {count} pairs into {}: truncated A={trunc_a}, B={trunc_b}",
                    args.out.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }

        Commands::Score(args) => {
            let suite = SuiteSpec::load_from_file(&args.suite)?;
            let case = suite.get_case(&args.case)?;
            let pairs_path = args
                .pairs
                .unwrap_or_else(|| PathBuf::from(&case.pairs_path));
            let records = load_pairs(&pairs_path)?;

            let report = evaluate_case(case, &records, args.bootstrap_iterations, args.seed)?;

            if !report.passed && bell {
                ring_terminal_bell();
            }

            if is_json {
                let status = if report.passed {
                    EnvelopeStatus::Success
                } else {
                    EnvelopeStatus::Error
                };
                let summary = format!(
                    "Case '{}': {} (Consistency {:.1}%, Kappa {:.3}, LC-WinRate {:.1}%)",
                    case.id,
                    if report.passed { "PASSED" } else { "FAILED" },
                    report.swap_metrics.swap_consistency_rate * 100.0,
                    report.swap_metrics.swap_kappa,
                    report.verbosity.length_controlled_win_rate_a * 100.0
                );
                let env = HarnessEnvelope {
                    status,
                    summary,
                    next_actions: report.failure_reasons.clone(),
                    artifacts: vec![pairs_path.display().to_string()],
                    data: report.clone(),
                };
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!("Case: {}", report.case_id);
                println!("Passed: {}", report.passed);
                println!("Regime: {:?}", report.regime);
                println!(
                    "Swap Consistency: {:.2}%",
                    report.swap_metrics.swap_consistency_rate * 100.0
                );
                println!("Flip Rate: {:.2}%", report.swap_metrics.flip_rate * 100.0);
                println!(
                    "Position Bias Index: {:.3}",
                    report.swap_metrics.position_bias_index
                );
                println!(
                    "Swap Agreement Kappa: {:.3}",
                    report.swap_metrics.swap_kappa
                );
                println!(
                    "Consistent Win Rate (A): {:.2}%",
                    report.swap_metrics.strict_win_rate_a * 100.0
                );
                println!(
                    "Length-Controlled Win Rate (A): {:.2}% (95% CI: [{:.2}%, {:.2}%])",
                    report.verbosity.length_controlled_win_rate_a * 100.0,
                    report.verbosity.lc_bootstrap_95_ci.0 * 100.0,
                    report.verbosity.lc_bootstrap_95_ci.1 * 100.0
                );
                println!(
                    "Verbosity Correlation r: {:.3}",
                    report.verbosity.length_correlation_r
                );
                if !report.failure_reasons.is_empty() {
                    println!("Violations:");
                    for reason in &report.failure_reasons {
                        println!("  - {reason}");
                    }
                }
            }

            if report.passed {
                Ok(ExitCode::SUCCESS)
            } else {
                Ok(ExitCode::from(1))
            }
        }

        Commands::Check(args) => {
            let suite = SuiteSpec::load_from_file(&args.suite)?;

            if let Some(lock_path) = &args.lock {
                verify_lockfile(lock_path, &args.suite, &args.root)?;
            }

            let mut all_reports = Vec::new();
            let mut any_failed = false;

            for case in &suite.cases {
                let dataset_path = args.root.join(&case.pairs_path);
                let records = load_pairs(&dataset_path)?;
                let rep = evaluate_case(case, &records, args.bootstrap_iterations, args.seed)?;
                if !rep.passed {
                    any_failed = true;
                }
                all_reports.push(rep);
            }

            if any_failed && bell {
                ring_terminal_bell();
            }

            if is_json {
                let status = if !any_failed {
                    EnvelopeStatus::Success
                } else {
                    EnvelopeStatus::Error
                };
                let summary = format!(
                    "Checked {} cases across suite '{}': {}",
                    all_reports.len(),
                    suite.suite.name,
                    if any_failed {
                        "Gating failures detected"
                    } else {
                        "All cases passed"
                    }
                );
                let env = HarnessEnvelope {
                    status,
                    summary,
                    next_actions: if any_failed {
                        vec!["Inspect failing cases and quarantine position flips".to_string()]
                    } else {
                        vec![]
                    },
                    artifacts: vec![args.suite.display().to_string()],
                    data: all_reports,
                };
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                for r in &all_reports {
                    println!(
                        "Case {:<32} | Passed: {:<5} | Consistency: {:5.1}% | LC-WR: {:5.1}%",
                        r.case_id,
                        r.passed,
                        r.swap_metrics.swap_consistency_rate * 100.0,
                        r.verbosity.length_controlled_win_rate_a * 100.0
                    );
                }
            }

            if any_failed {
                Ok(ExitCode::from(1))
            } else {
                Ok(ExitCode::SUCCESS)
            }
        }

        Commands::Compare(args) => {
            let suite = SuiteSpec::load_from_file(&args.suite)?;
            let case = suite.get_case(&args.case)?;

            let recs_base = load_pairs(&args.baseline)?;
            let recs_cand = load_pairs(&args.candidate)?;

            let rep_base = evaluate_case(case, &recs_base, 0, 42)?;
            let rep_cand = evaluate_case(case, &recs_cand, 0, 42)?;

            let delta_consistency = rep_cand.swap_metrics.swap_consistency_rate
                - rep_base.swap_metrics.swap_consistency_rate;
            let delta_lc_wr = rep_cand.verbosity.length_controlled_win_rate_a
                - rep_base.verbosity.length_controlled_win_rate_a;

            if is_json {
                let env = HarnessEnvelope::success(
                    format!(
                        "Compared baseline vs candidate: Delta Consistency: {:.2}%, Delta LC-WinRate: {:.2}%",
                        delta_consistency * 100.0,
                        delta_lc_wr * 100.0
                    ),
                    vec![
                        args.baseline.display().to_string(),
                        args.candidate.display().to_string(),
                    ],
                    serde_json::json!({
                        "baseline": rep_base,
                        "candidate": rep_cand,
                        "delta_swap_consistency": delta_consistency,
                        "delta_lc_win_rate": delta_lc_wr,
                    }),
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!(
                    "Baseline Consistency:  {:.2}%",
                    rep_base.swap_metrics.swap_consistency_rate * 100.0
                );
                println!(
                    "Candidate Consistency: {:.2}%",
                    rep_cand.swap_metrics.swap_consistency_rate * 100.0
                );
                println!("Delta Consistency:     {:+.2}%", delta_consistency * 100.0);
                println!(
                    "Baseline LC-WinRate:   {:.2}%",
                    rep_base.verbosity.length_controlled_win_rate_a * 100.0
                );
                println!(
                    "Candidate LC-WinRate:  {:.2}%",
                    rep_cand.verbosity.length_controlled_win_rate_a * 100.0
                );
                println!("Delta LC-WinRate:      {:+.2}%", delta_lc_wr * 100.0);
            }

            Ok(ExitCode::SUCCESS)
        }

        Commands::Rank(args) => {
            let records = load_pairs(&args.pairs)?;
            let tourney = rank_tournament(&records, args.bootstrap_iterations, args.seed);

            if is_json {
                let env = HarnessEnvelope::success(
                    format!(
                        "Ranked {} candidates across {} pairs (consistent: {}, flips: {})",
                        tourney.ratings.len(),
                        tourney.total_pairs,
                        tourney.consistent_pairs,
                        tourney.quarantined_flips
                    ),
                    vec![args.pairs.display().to_string()],
                    tourney,
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!(
                    "Tournament: {} total pairs, {} consistent, {} quarantined flips\n",
                    tourney.total_pairs, tourney.consistent_pairs, tourney.quarantined_flips
                );
                println!(
                    "{:<24} | {:<9} | {:<10} | {:<8} | 95% Bootstrap CI",
                    "Candidate", "Elo", "Win Rate", "BT-Prob"
                );
                println!("{:-<75}", "");
                for r in &tourney.ratings {
                    println!(
                        "{:<24} | {:>9.1} | {:>9.1}% | {:>8.4} | [{:.1}, {:.1}]",
                        r.candidate_id,
                        r.elo_rating,
                        r.consistent_win_rate * 100.0,
                        r.bradley_terry_strength,
                        r.elo_bootstrap_95_ci.0,
                        r.elo_bootstrap_95_ci.1
                    );
                }
            }

            Ok(ExitCode::SUCCESS)
        }

        Commands::Simulate(args) => {
            let suite = SuiteSpec::load_from_file(&args.suite)?;
            let case = suite.get_case(&args.case)?;
            let pairs_path = PathBuf::from(&case.pairs_path);
            let raw_records = load_pairs(&pairs_path)?;

            let mut prng = SplitMix64::new(args.seed);
            let mut sim_records = Vec::with_capacity(raw_records.len());

            for rec in &raw_records {
                let (f_j, s_j) = simulate_pair_judgments(
                    &rec.pair_id,
                    &case.candidate_a,
                    &case.candidate_b,
                    &rec.prompt,
                    &rec.candidate_a_response,
                    &rec.candidate_b_response,
                    args.profile,
                    &mut prng,
                );

                let mut s_rec = rec.clone();
                s_rec.forward_pass = f_j;
                s_rec.swapped_pass = s_j;
                sim_records.push(s_rec);
            }

            let rep = evaluate_case(case, &sim_records, 0, args.seed)?;

            if let Some(out_path) = args.out {
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut f = File::create(&out_path)?;
                for r in &sim_records {
                    writeln!(f, "{}", serde_json::to_string(r)?)?;
                }
            }

            if is_json {
                let env = HarnessEnvelope::success(
                    format!(
                        "Simulated profile {:?} over {} pairs",
                        args.profile,
                        sim_records.len()
                    ),
                    vec![],
                    rep,
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!("Simulated Profile: {:?}", args.profile);
                println!("Outcome Regime: {:?}", rep.regime);
                println!(
                    "Swap Consistency: {:.2}%",
                    rep.swap_metrics.swap_consistency_rate * 100.0
                );
                println!(
                    "Position Bias Index: {:.3}",
                    rep.swap_metrics.position_bias_index
                );
            }

            Ok(ExitCode::SUCCESS)
        }

        Commands::Lock(args) => {
            let lockfile = generate_lockfile(&args.suite, &args.root)?;
            let out_path = args.out.unwrap_or_else(|| {
                args.suite
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("softfactory.lock.json")
            });

            fs::write(&out_path, serde_json::to_string_pretty(&lockfile)?)?;

            if is_json {
                let env = HarnessEnvelope::success(
                    format!("Generated lockfile at {}", out_path.display()),
                    vec![out_path.display().to_string()],
                    lockfile,
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!(
                    "Generated sha256-framed-v1 lockfile at {} (digest {})",
                    out_path.display(),
                    lockfile.overall_lock_digest
                );
            }
            Ok(ExitCode::SUCCESS)
        }

        Commands::Verify(args) => {
            verify_lockfile(&args.lock, &args.suite, &args.root)?;

            if is_json {
                let env = HarnessEnvelope::success(
                    format!("Verified lockfile {}", args.lock.display()),
                    vec![args.lock.display().to_string()],
                    (),
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!("Lockfile verification passed: zero drift detected");
            }
            Ok(ExitCode::SUCCESS)
        }

        Commands::Export(args) => {
            let suite = SuiteSpec::load_from_file(&args.suite)?;
            let mut reports = Vec::new();

            for case in &suite.cases {
                let dataset_path = args.root.join(&case.pairs_path);
                if dataset_path.exists() {
                    let records = load_pairs(&dataset_path)?;
                    let rep = evaluate_case(case, &records, 0, 42)?;
                    reports.push(rep);
                }
            }

            let rendered = match args.format.to_lowercase().as_str() {
                "mermaid" => export_mermaid_flowchart(&reports),
                "otel" => serde_json::to_string_pretty(&export_otel_spans(&reports, &suite))?,
                _ => anyhow::bail!("unknown export format '{}'", args.format),
            };

            if let Some(out_path) = args.out {
                fs::write(&out_path, &rendered)?;
                if is_json {
                    let env = HarnessEnvelope::success(
                        format!("Exported {} format to {}", args.format, out_path.display()),
                        vec![out_path.display().to_string()],
                        (),
                    );
                    println!("{}", serde_json::to_string_pretty(&env)?);
                } else {
                    println!("Exported {} to {}", args.format, out_path.display());
                }
            } else {
                println!("{rendered}");
            }

            Ok(ExitCode::SUCCESS)
        }

        Commands::Ingest(args) => {
            let ingested = poise::ingest::ingest_agent_harness_runs(
                &args.runs_dir,
                &args.candidate_a,
                &args.candidate_b,
            )?;

            if is_json {
                let env = HarnessEnvelope::success(
                    format!(
                        "Ingested {} pairs from {}",
                        ingested.len(),
                        args.runs_dir.display()
                    ),
                    vec![],
                    ingested,
                );
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!(
                    "Ingested {} pairs comparing {} vs {} from {}",
                    ingested.len(),
                    args.candidate_a,
                    args.candidate_b,
                    args.runs_dir.display()
                );
            }

            Ok(ExitCode::SUCCESS)
        }

        Commands::Repo(args) => {
            let rep = audit_repository(&args.path);

            if !rep.passed && bell {
                ring_terminal_bell();
            }

            if is_json {
                let status = if rep.passed {
                    EnvelopeStatus::Success
                } else {
                    EnvelopeStatus::Error
                };
                let env = HarnessEnvelope {
                    status,
                    summary: format!(
                        "Repo audit at {}: {}/{} gates passed",
                        args.path.display(),
                        rep.passed_gates,
                        rep.total_gates
                    ),
                    next_actions: if rep.passed {
                        vec![]
                    } else {
                        vec!["Resolve failed repository governance gates".to_string()]
                    },
                    artifacts: vec![args.path.display().to_string()],
                    data: rep.clone(),
                };
                println!("{}", serde_json::to_string_pretty(&env)?);
            } else {
                println!(
                    "Repo Audit: {} (Passed {}/{} gates)",
                    if rep.passed { "PASSED" } else { "FAILED" },
                    rep.passed_gates,
                    rep.total_gates
                );
                for g in &rep.gates {
                    let status_str = if g.passed { "PASS" } else { "FAIL" };
                    println!("  [{status_str}] {:<42} : {}", g.name, g.details);
                }
            }

            if rep.passed {
                Ok(ExitCode::SUCCESS)
            } else {
                Ok(ExitCode::from(1))
            }
        }
    }
}
