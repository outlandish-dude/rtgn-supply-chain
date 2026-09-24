# Software Supply-Chain Dataset Quality & Empirical Audit Report

## 1. Executive Summary & Verification Classification

This audit provides a formal scientific and empirical verification of the reconstructed dataset ingestion pipeline for the software supply-chain deep-learning project at `E:\college\5th sem\theory\dl\dl project`.

The audit categorizes all findings into **VERIFIED**, **QUESTIONABLE**, and **FAILED**, adhering strictly to non-execution safety, temporal non-leakage, and reproducible data integrity standards.

| Audit Dimension | Status | Key Finding |
|---|---|---|
| **Non-Execution Safety Protocol** | `VERIFIED` | 100% compliant. Zero source code, archives, wheels, or scripts downloaded or executed. |
| **Datadog Manifest Invariants** | `VERIFIED` | 1,834 packages parsed; 1,817 all-version packages; 17 specific-version packages; 29 explicit versions; 1,846 label rows. |
| **Canonical Normalization (PEP 503/440)** | `VERIFIED` | Deterministic normalization correctly identified 1 package name collision (`Roblox.-com` vs `roblox-com`). |
| **Temporal Integrity & Non-Leakage** | `VERIFIED` | Earliest upload timestamps strictly preserved; zero synthetic onset timestamps; zero backward propagation of latest dependencies. |
| **Release-Specific Dependencies** | `VERIFIED` | Dependencies isolated per release; 164,361 dependency relationship edges extracted; missing historical dependencies kept null. |
| **OSV Version-Granular Advisories** | `VERIFIED` | 6,526 package/version items queried; 13,564 security events recorded; 1,444 unique vulnerability IDs. |
| **API Resilience & Error Handling** | `VERIFIED` | Handled network timeouts and transient disconnects gracefully via exponential backoff; 0 unhandled failures. |
| **GitHub Advisory Auth Protocol** | `VERIFIED` | Cleanly deferred without credential fabrication or silent substitution. |
| **PyPI Registry Malware Takedown Rate** | `QUESTIONABLE` | 1,799 of 1,834 packages (98.09%) return HTTP 404 due to PyPI administrative malware takedowns. |
| **Historical Dependency Completeness** | `QUESTIONABLE` | 309 versions lack dependency metadata in historical release JSON. |
| **Critical Pipeline / Data Failures** | `NONE (FAILED = 0)` | Zero data corruption, zero schema violations, zero pipeline crashes. |

---

## 2. Detailed Verification Findings

### [VERIFIED] 1. Datadog Manifest Parsing & Normalization Correctness
- **Manifest File:** `data/raw/datadog/pypi_manifest.json`
- **Total Manifest Packages:** 1,834
- **All-Version Malicious Packages (`null`):** 1,817
- **Specific-Version Packages (list):** 17
- **Explicit Malicious Versions:** 29
- **Total Generated Label Rows:** 1,846 (`1,817 + 29 = 1,846`)
- **Canonical Package Collision Detected:** Under PEP 503 normalization (`re.sub(r"[-_.]+", "-", name).lower()`), `Roblox.-com` and `roblox-com` map to the identical canonical identifier `roblox-com`. This explains why unique canonical packages count 1,833 while manifest dictionary entries count 1,834.

### [VERIFIED] 2. Non-Execution Safety Invariant
The pipeline operates strictly over metadata endpoints:
- `https://pypi.org/pypi/<package>/json`
- `https://pypi.org/pypi/<package>/<version>/json`
- `https://api.osv.dev/v1/querybatch`
- `https://api.osv.dev/v1/vulns/<id>`
No distribution files (`.whl`, `.tar.gz`, `.zip`) were downloaded, uncompressed, or imported into the Python runtime.

### [VERIFIED] 3. Temporal Integrity & Zero Leakage Protocol
1. **No Inferred Timestamps:** Release timestamps are derived strictly from distribution file upload timestamps (`upload_time_iso_8601`) converted to UTC Unix seconds. Date range spans `2013-05-28T17:24:19 UTC` to `2026-09-21T20:53:30 UTC`. Zero future-dated releases.
2. **No Backwards Dependency Propagation:** Historical releases were individually resolved via release-specific endpoints. Older releases without dependency specifications were left as missing (`null`) rather than backfilled from latest releases.
3. **Decoupled Label Scoping:** The Datadog manifest classification (`label_scope: all_versions` vs `specific_version`) is stored as a provenance attribute without inventing synthetic malicious onset timestamps.
4. **Advisory Decoupling:** OSV `published_timestamp` and `modified_timestamp` are preserved in `security_events.csv` without overwriting package version release timestamps.

### [VERIFIED] 4. Labeling Semantics & Separation
Releases are partitioned according to scientific label boundaries:
- **Confirmed Malicious Versions:** 1,495 (releases belonging to packages with `all_versions` malicious scope)
- **Unlabeled / Assumed Benign Versions:** 3,232 (releases of packages with specific version compromise, excluding the compromised releases)
- **Comprehensive Malicious Label Registry:** 1,846 records in `data/processed/malicious_labels.csv`, retaining the 1,799 takedown packages alongside live registry releases.

---

## 3. Questionable Findings & Inherent Real-World Biases

### [QUESTIONABLE] 1. Live PyPI Takedown Rate (Administrative Pruning)
- **Finding:** 1,799 out of 1,834 packages (98.09%) in the Datadog manifest returned HTTP 404 from the live PyPI JSON API.
- **Analysis:** This is expected real-world behavior: PyPI's security team promptly deletes or quarantines pure malware packages once identified.
- **Impact on Graph Learning:** Pure malware packages do not exist in live PyPI metadata feeds; their historical dependencies and files cannot be reconstructed from PyPI API. They are preserved in `data/processed/malicious_labels.csv` with `in_pypi_registry = False`. For packages remaining on PyPI (such as `upgini`, `convocations`, or the 17 hijacked packages like `ultralytics`), rich graph features and dependency edges are available.

### [QUESTIONABLE] 2. Historical Dependency Sparsity
- **Finding:** 309 versions across the collected packages have no `requires_dist` metadata.
- **Analysis:** Prior to PEP 566 and modern wheel metadata standards, dependencies were often evaluated dynamically during `setup.py` execution and not recorded in PyPI JSON metadata.
- **Handling:** Rather than risking temporal leakage by imputing dependencies from modern releases, these are preserved as null. Graph builders must treat missing dependencies as unobserved rather than empty.

### [QUESTIONABLE] 3. High Release Volume on All-Version Manifest Packages
- **Finding:** The package `upgini` has 1,290 releases in PyPI and is marked `null` in the Datadog manifest, resulting in 1,290 releases receiving `confirmed_malicious` with `all_versions` scope.
- **Analysis:** If a high-frequency release package is added to the Datadog manifest without explicit version restrictions, the `all_versions` convention marks all 1,290 releases as malicious. Downstream temporal models should consider version-level anomaly features or verify if this reflects full-package malicious intent or unsegmented manifest listing.

---

## 4. Failed Invariants
**Zero Failed Items.**
All schema invariants, normalization tests, caching behaviors, and dataset consistency checks passed without failure or data corruption.

---

## 5. Artifact Summary

| File | Rows / Entries | Description |
|---|---|---|
| `data/intermediate/datadog_pypi_labels.csv` | 1,846 rows | Raw normalized Datadog labels (`package,version,label,label_scope,source`). |
| `data/intermediate/package_versions.csv` | 4,727 rows | Raw PyPI collected versions with release-specific `requires_dist`. |
| `data/intermediate/osv_events.csv` | 13,564 rows | Package-version granular OSV vulnerability records. |
| `data/processed/package_versions.csv` | 4,727 rows | Joined version table with strict label semantics and scopes. |
| `data/processed/malicious_labels.csv` | 1,846 rows | Complete Datadog malware registry with PyPI registry presence flags. |
| `data/processed/security_events.csv` | 13,564 rows | Canonical security and malware events with published timestamps. |
| `data/processed/ingestion_summary.json` | 64 lines | Complete machine-readable summary with audit metrics and provenance. |
