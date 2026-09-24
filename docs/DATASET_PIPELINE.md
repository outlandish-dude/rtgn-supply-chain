# Software Supply-Chain Dataset Ingestion Pipeline Specification

## 1. Overview & Research Objective
This document specifies the architecture, protocols, and data schemas for the empirical dataset ingestion pipeline supporting the research on **Temporal Heterogeneous Graph Learning for Software Supply-Chain Compromise Detection**.

The dataset transitions our study from an initial 5-sample prototype to a comprehensive, empirical supply-chain security dataset derived from the curated **Datadog PyPI Malware Manifest** (1,834 packages), cross-referenced with live PyPI registry metadata, release histories, dependency graphs, and Open Source Vulnerabilities (OSV) records.

---

## 2. Fundamental Safety Protocol

> [!CAUTION]
> **STRICT NON-EXECUTION INVARIANT:**
> Under no circumstances does this pipeline download, extract, install, execute, or import raw package source code, wheels, setup scripts, or binaries from any suspected or confirmed malicious packages.
> 
> All ingestion operates **strictly on metadata**:
> - JSON release manifests
> - Upload timestamps
> - Dependency specification strings (`requires_dist`)
> - Project and VCS URLs
> - Security advisories and vulnerability identifiers
> 
> No untrusted Python code is ever executed or evaluated.

---

## 3. Data Sources & API Endpoints

| Source Name | Ingestion Type | Endpoint / Path | Authentication | Caching Policy |
|---|---|---|---|---|
| **Datadog Manifest** | Local JSON File | `data/raw/datadog/pypi_manifest.json` | None (Local file) | Immutable raw file |
| **PyPI Package Metadata** | REST HTTP GET | `https://pypi.org/pypi/{package}/json` | None (Public) | Persistent disk cache in `data/raw/pypi/pkg_{name}.json`. HTTP 404 responses are negatively cached to prevent redundant queries. |
| **PyPI Release Metadata** | REST HTTP GET | `https://pypi.org/pypi/{package}/{version}/json` | None (Public) | Persistent disk cache in `data/raw/pypi/rel_{pkg}_{ver}.json`. Used strictly for release-specific metadata. |
| **OSV QueryBatch** | REST HTTP POST | `https://api.osv.dev/v1/querybatch` | None (Public) | Batched requests (500 queries/batch). |
| **OSV Vulnerability Details** | REST HTTP GET | `https://api.osv.dev/v1/vulns/{id}` | None (Public) | Persistent disk cache in `data/raw/osv/vuln_{id}.json`. |
| **GitHub Advisory Database** | GraphQL / REST | `https://api.github.com/graphql` | `GITHUB_TOKEN` required | Unauthenticated calls are explicitly deferred per Rule 10 without credential fabrication or silent substitution. |

---

## 4. Canonical Normalization Rules

Deterministic normalization prevents multi-relational entity collisions:

### Package Names (PEP 503)
Following the Python Enhancement Proposal 503 standard:
```python
def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()
```
*Example:* `Roblox.-com` and `roblox-com` both normalize to `roblox-com`, resolving cross-source casing and separator variations.

### Versions (PEP 440)
Versions are parsed using `packaging.version.Version`:
- Valid PEP 440 versions are parsed to their canonical string representation.
- Non-standard versions retain their raw string representation and are flagged in quality audits.
- The special marker `__ALL__` denotes all versions of a package.

### Repository URLs
Git and GitHub repository URLs are stripped of protocol prefixes (`git+`, `http://`), trailing `.git` extensions, and normalized to canonical `github:{owner}/{repo}` format.

---

## 5. Label Semantics & Scoping Hierarchy

The dataset maintains a strict labeling hierarchy:

### Datadog Manifest Interpretation
1. **`null` Entry:**
   - Signifies the package was created with malicious intent or all versions are considered compromised.
   - `label = "confirmed_malicious"`
   - `label_scope = "all_versions"`
   - When joined onto PyPI releases: every release of this package is tagged `confirmed_malicious`.
2. **Explicit Version List:**
   - Signifies specific release compromise (e.g. account takeover, trojan update, dependency hijacking).
   - `label = "confirmed_malicious"`
   - `label_scope = "specific_version"`
   - When joined onto PyPI releases: **only** the exact matching versions are tagged `confirmed_malicious`.
3. **Unlabeled Releases:**
   - Any release not covered by a confirmed malicious rule is classified as `unlabeled / assumed_benign` (`label_scope = "unlabeled"`).
   - Releases are **never** assumed to be "ground-truth benign" without independent empirical verification.

---

## 6. Strict Temporal Non-Leakage Protocols

To ensure scientific validity in temporal graph learning and early-warning compromise detection, the pipeline enforces strict temporal integrity:

1. **Release-Specific Dependency Isolation:**
   - Dependencies (`requires_dist`) are extracted strictly from release-specific endpoints (`/pypi/{package}/{version}/json`).
   - Project-level `info.requires_dist` from the latest release is **never** propagated backward to historical releases.
   - Historical releases without metadata explicitly preserve `requires_dist` as missing (`null`).
2. **Preservation of Authentic Timestamps:**
   - Release timestamps reflect the earliest distribution file upload timestamp (`upload_time_iso_8601`) converted to UTC Unix seconds.
   - Advisory timestamps reflect public disclosure dates (`published`) or modification dates (`modified`).
   - Advisory timestamps are **never** used to overwrite release timestamps.
   - No synthetic or inferred "malicious onset" timestamps are fabricated.
3. **Decoupling of Scopes and Timestamps:**
   - `label_scope` is recorded as a static provenance attribute, completely independent of event timestamps.

---

## 7. Data Artifact Schemas

### `data/intermediate/datadog_pypi_labels.csv`
- `package` (string): Raw package name from manifest.
- `version` (string): Specific version string or `__ALL__`.
- `label` (string): `confirmed_malicious`.
- `label_scope` (string): `all_versions` or `specific_version`.
- `source` (string): `datadog`.

### `data/intermediate/package_versions.csv`
- `package` (string): Raw package name from PyPI.
- `normalized_package` (string): PEP 503 normalized name.
- `version` (string): Release version string.
- `release_timestamp` (float): Unix epoch timestamp of earliest upload.
- `requires_python` (string): Python version specification.
- `requires_dist` (string): JSON list of dependency requirement strings.
- `repository_url` (string): Canonical repository identifier.
- `homepage_url` (string): Homepage URL.
- `source` (string): `pypi`.

### `data/intermediate/osv_events.csv`
- `osv_id` (string): Advisory identifier (e.g. `MAL-2024-XXXX`, `GHSA-...`).
- `package` (string): Package name.
- `version` (string): Target version queried or `__ALL__`.
- `published_timestamp` (float): Advisory publication epoch timestamp.
- `modified_timestamp` (float): Advisory last-modified epoch timestamp.
- `affected_ranges` (string): JSON array of affected version ranges.
- `summary` (string): Vulnerability summary description.
- `references` (string): JSON array of reference URLs.
- `source` (string): `osv`.

### `data/processed/package_versions.csv`
Includes all collected PyPI versions with joined Datadog labels:
- `package`, `normalized_package`, `version`, `release_timestamp`
- `label` (`confirmed_malicious` | `unlabeled / assumed_benign`)
- `label_scope` (`all_versions` | `specific_version` | `unlabeled`)
- `label_source` (`datadog` | `none`)
- `requires_python`, `requires_dist`, `repository_url`, `homepage_url`, `source`

### `data/processed/malicious_labels.csv`
Comprehensive ground-truth registry of all 1,846 Datadog malicious label rows:
- `package`, `normalized_package`, `version`, `label`, `label_scope`, `source`, `in_pypi_registry`

### `data/processed/security_events.csv`
Canonical security events from OSV and malware advisories:
- `event_id`, `event_type`, `package`, `normalized_package`, `version`, `advisory_id`, `published_timestamp`, `modified_timestamp`, `summary`, `affected_ranges`, `references`, `source`

### `data/processed/ingestion_summary.json`
Machine-readable audit manifest containing exact counts, date ranges, API failure audits, and missing metadata statistics.

---

## 8. Reproducibility & Pipeline Execution Instructions

### Prerequisites
- Python 3.11 or 3.12
- Required packages: `requests`, `packaging`, `pydantic`, `pytest`

### Step 1: Validate Datadog Manifest
```powershell
python scripts/validate_datadog_manifest.py
```
Validates manifest invariants and exports `data/intermediate/datadog_pypi_labels.csv`.

### Step 2: Run Automated Test Suite
```powershell
pytest tests/
```
Runs 9 comprehensive unit tests verifying parsing, caching, negative 404 caching, version-specific dependency isolation, and label joining logic.

### Step 3: Run Full Reproducible Ingestion Pipeline
```powershell
python scripts/run_reproducible_pipeline.py
```
Orchestrates end-to-end ingestion, populates intermediate caches, performs joins, and outputs processed datasets and `ingestion_summary.json`.
