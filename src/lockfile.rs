use crate::spec::SuiteSpec;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LockfileError {
    #[error("failed to read file {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse lockfile JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("lockfile drift detected: {0}")]
    Drift(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseLockEntry {
    pub case_id: String,
    pub stratum: String,
    pub dataset_sha256: String,
    pub spec_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoiseLockfile {
    pub lock_version: String,
    pub suite_name: String,
    pub suite_version: String,
    pub suite_digest: String,
    pub cases: BTreeMap<String, CaseLockEntry>,
    pub overall_lock_digest: String,
}

pub fn hash_file_framed(path: impl AsRef<Path>) -> Result<String, LockfileError> {
    let p = path.as_ref();
    let bytes = fs::read(p).map_err(|e| LockfileError::Io {
        path: p.display().to_string(),
        source: e,
    })?;

    let mut hasher = Sha256::new();
    hasher.update(b"tideos-poise-file-v1\0");
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn generate_lockfile(
    suite_path: impl AsRef<Path>,
    root_dir: impl AsRef<Path>,
) -> Result<PoiseLockfile, LockfileError> {
    let sp = suite_path.as_ref();
    let root = root_dir.as_ref();

    let suite = SuiteSpec::load_from_file(sp).map_err(|e| LockfileError::Drift(e.to_string()))?;
    let suite_digest = hash_file_framed(sp)?;

    let mut cases = BTreeMap::new();
    let mut overall_hasher = Sha256::new();
    overall_hasher.update(b"tideos-poise-lock-v1\0");
    overall_hasher.update(suite_digest.as_bytes());

    for case in &suite.cases {
        let dataset_full_path = root.join(&case.pairs_path);
        let dataset_digest = if dataset_full_path.exists() {
            hash_file_framed(&dataset_full_path)?
        } else {
            "missing-dataset".to_string()
        };

        let mut spec_hasher = Sha256::new();
        spec_hasher.update(case.id.as_bytes());
        spec_hasher.update(case.pinned_model.as_bytes());
        spec_hasher.update(case.min_swap_consistency.to_be_bytes());
        spec_hasher.update(case.max_flip_rate.to_be_bytes());
        spec_hasher.update(case.length_policy.max_tokens.to_be_bytes());
        let spec_digest = format!("{:x}", spec_hasher.finalize());

        overall_hasher.update(case.id.as_bytes());
        overall_hasher.update(dataset_digest.as_bytes());
        overall_hasher.update(spec_digest.as_bytes());

        cases.insert(
            case.id.clone(),
            CaseLockEntry {
                case_id: case.id.clone(),
                stratum: case.stratum.clone(),
                dataset_sha256: dataset_digest,
                spec_sha256: spec_digest,
            },
        );
    }

    let overall_digest = format!("{:x}", overall_hasher.finalize());

    Ok(PoiseLockfile {
        lock_version: "sha256-framed-v1".to_string(),
        suite_name: suite.suite.name,
        suite_version: suite.suite.version,
        suite_digest,
        cases,
        overall_lock_digest: overall_digest,
    })
}

pub fn verify_lockfile(
    lockfile_path: impl AsRef<Path>,
    suite_path: impl AsRef<Path>,
    root_dir: impl AsRef<Path>,
) -> Result<(), LockfileError> {
    let lp = lockfile_path.as_ref();
    let content = fs::read_to_string(lp).map_err(|e| LockfileError::Io {
        path: lp.display().to_string(),
        source: e,
    })?;
    let existing_lock: PoiseLockfile = serde_json::from_str(&content)?;

    let current_lock = generate_lockfile(suite_path, root_dir)?;

    if existing_lock.overall_lock_digest != current_lock.overall_lock_digest {
        return Err(LockfileError::Drift(format!(
            "overall lock digest mismatch: locked '{}' != current '{}'",
            existing_lock.overall_lock_digest, current_lock.overall_lock_digest
        )));
    }

    for (id, existing_case) in &existing_lock.cases {
        let current_case = current_lock.cases.get(id).ok_or_else(|| {
            LockfileError::Drift(format!(
                "case '{id}' present in lockfile but missing in suite"
            ))
        })?;

        if existing_case.dataset_sha256 != current_case.dataset_sha256 {
            return Err(LockfileError::Drift(format!(
                "case '{id}' dataset hash drift: locked '{}' != current '{}'",
                existing_case.dataset_sha256, current_case.dataset_sha256
            )));
        }

        if existing_case.spec_sha256 != current_case.spec_sha256 {
            return Err(LockfileError::Drift(format!(
                "case '{id}' spec parameters drift: locked '{}' != current '{}'",
                existing_case.spec_sha256, current_case.spec_sha256
            )));
        }
    }

    Ok(())
}
