use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestedPairRecord {
    pub run_id_a: String,
    pub run_id_b: String,
    pub prompt: String,
    pub candidate_a: String,
    pub candidate_b: String,
    pub response_a: String,
    pub response_b: String,
}

pub fn ingest_agent_harness_runs(
    runs_dir: impl AsRef<Path>,
    candidate_a: &str,
    candidate_b: &str,
) -> Result<Vec<IngestedPairRecord>> {
    let p = runs_dir.as_ref();
    if !p.exists() {
        return Ok(Vec::new());
    }

    let mut candidate_a_runs = Vec::new();
    let mut candidate_b_runs = Vec::new();

    for entry in fs::read_dir(p).context("failed to read harness runs directory")? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
        if dir_name.contains(candidate_a) {
            candidate_a_runs.push(path);
        } else if dir_name.contains(candidate_b) {
            candidate_b_runs.push(path);
        }
    }

    let mut pairs = Vec::new();
    let min_len = candidate_a_runs.len().min(candidate_b_runs.len());

    for i in 0..min_len {
        let path_a = &candidate_a_runs[i];
        let path_b = &candidate_b_runs[i];

        let prompt_path = path_a.join("prompt.md");
        let prompt = if prompt_path.exists() {
            fs::read_to_string(prompt_path).unwrap_or_default()
        } else {
            "Agent task prompt".to_string()
        };

        let stdout_a = path_a.join("stdout.log");
        let stdout_b = path_b.join("stdout.log");

        let response_a = if stdout_a.exists() {
            fs::read_to_string(stdout_a).unwrap_or_default()
        } else {
            "Agent A stdout".to_string()
        };

        let response_b = if stdout_b.exists() {
            fs::read_to_string(stdout_b).unwrap_or_default()
        } else {
            "Agent B stdout".to_string()
        };

        pairs.push(IngestedPairRecord {
            run_id_a: path_a
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            run_id_b: path_b
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            prompt,
            candidate_a: candidate_a.to_string(),
            candidate_b: candidate_b.to_string(),
            response_a,
            response_b,
        });
    }

    Ok(pairs)
}
