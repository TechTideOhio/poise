use crate::adjudicator::CaseAdjudicationReport;
use crate::spec::SuiteSpec;
use serde_json::json;

pub fn render_reference_table(
    true_win_rate_a: f64,
    primacy_bias: f64,
    verbosity_skew: f64,
) -> String {
    let mut out = String::new();
    out.push_str("\n+---------------------------------------------------------------------------------------------------------+\n");
    out.push_str("|           PAIRWISE LLM JUDGE POSITION BIAS & VERBOSITY SKEW DISTORTION (poise Tool 10)                 |\n");
    out.push_str("+---------------------------------------------------------------------------------------------------------+\n");
    out.push_str("| Metric / Scenario             | Naive Single-Pass A/B | Double-Blind Swapped | Poise LC-WinRate Corrected |\n");
    out.push_str("+-------------------------------+-----------------------+----------------------+----------------------------+\n");

    let naive_primacy = (true_win_rate_a + primacy_bias).clamp(0.0, 1.0);
    let naive_verbosity = (true_win_rate_a + verbosity_skew).clamp(0.0, 1.0);
    let naive_compound = (true_win_rate_a + primacy_bias + verbosity_skew).clamp(0.0, 1.0);

    out.push_str(&format!(
        "| Symmetric Fair Models (50/50) | {:5.1}% (Primacy +{:.0}%) | 50.0% (Flips Filtered)| 50.0% (Zero Length Bias)   |\n",
        naive_primacy * 100.0,
        primacy_bias * 100.0
    ));
    out.push_str(&format!(
        "| Padded Verbose Challenger     | {:5.1}% (Length +{:.0}%)  | {:5.1}% (Length-Skewed) | {:5.1}% (Debiased True)    |\n",
        naive_verbosity * 100.0,
        verbosity_skew * 100.0,
        naive_verbosity * 100.0,
        true_win_rate_a * 100.0
    ));
    out.push_str(&format!(
        "| Slot1 + Verbose Double Mirage | {:5.1}% (Distorted)   | {:5.1}% (Order Safe)   | {:5.1}% (True Capability)  |\n",
        naive_compound * 100.0,
        naive_verbosity * 100.0,
        true_win_rate_a * 100.0
    ));
    out.push_str("+-------------------------------+-----------------------+----------------------+----------------------------+\n");
    out.push_str("| Key Metrology Principle: In analytical assaying (Gauss 1820), swapping pans on a balance beam           |\n");
    out.push_str("| eliminates lever-arm asymmetry. Counting only consistent verdicts cancels LLM slot preference.           |\n");
    out.push_str("+---------------------------------------------------------------------------------------------------------+\n\n");
    out
}

pub fn export_mermaid_flowchart(reports: &[CaseAdjudicationReport]) -> String {
    let mut chart = String::from("flowchart LR\n");
    chart.push_str("    subgraph InputPairs [Double-Blind Evaluation Pairs]\n");
    chart.push_str("        P1[Pass 1: Forward Slot 1=A, Slot 2=B]\n");
    chart.push_str("        P2[Pass 2: Swapped Slot 1=B, Slot 2=A]\n");
    chart.push_str("    end\n");
    chart.push_str("    P1 --> AD[Poise Adjudicator: Consistent Verdict Filter]\n");
    chart.push_str("    P2 --> AD\n");
    chart.push_str("    subgraph Outcomes [Quarantine & Ratings]\n");
    chart.push_str("        AD -->|Consistent Win| CW[Consistent Wins A / B]\n");
    chart.push_str("        AD -->|Position Flip| QF[Quarantined Primacy/Recency Flips]\n");
    chart.push_str("    end\n");
    chart.push_str("    CW --> LC[Logistic LC-WinRate & Bradley-Terry Ratings]\n");

    for r in reports {
        let node_id = r.case_id.replace('-', "_");
        chart.push_str(&format!(
            "    CW -.-> {node_id}[\"{}: Consistency {:.1}%, LC-WR {:.1}%\"]\n",
            r.case_id,
            r.swap_metrics.swap_consistency_rate * 100.0,
            r.verbosity.length_controlled_win_rate_a * 100.0
        ));
    }

    chart
}

pub fn export_otel_spans(
    reports: &[CaseAdjudicationReport],
    suite: &SuiteSpec,
) -> serde_json::Value {
    let spans: Vec<serde_json::Value> = reports
        .iter()
        .map(|r| {
            json!({
                "trace_id": format!("tideos-poise-{}", suite.suite.name),
                "span_id": format!("case-{}", r.case_id),
                "name": "eval.pairwise.adjudicate",
                "attributes": {
                    "eval.case_id": r.case_id,
                    "eval.stratum": r.stratum,
                    "eval.regime": format!("{:?}", r.regime),
                    "eval.passed": r.passed,
                    "eval.swap_consistency_rate": r.swap_metrics.swap_consistency_rate,
                    "eval.flip_rate": r.swap_metrics.flip_rate,
                    "eval.position_bias_index": r.swap_metrics.position_bias_index,
                    "eval.swap_kappa": r.swap_metrics.swap_kappa,
                    "eval.raw_win_rate_a": r.verbosity.raw_win_rate_a,
                    "eval.length_controlled_win_rate_a": r.verbosity.length_controlled_win_rate_a,
                    "eval.length_correlation_r": r.verbosity.length_correlation_r,
                    "eval.verbosity_inflation_delta": r.verbosity.verbosity_inflation_delta,
                }
            })
        })
        .collect();

    json!({
        "resource": {
            "service.name": "poise-eval",
            "service.version": "0.1.0"
        },
        "spans": spans
    })
}
