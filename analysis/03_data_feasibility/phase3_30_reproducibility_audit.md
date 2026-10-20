# Phase 3.30 — End-to-end reproducibility audit

Audit date: 2026-10-20. Output hashes below use SHA-256. All rerun-generated files, if any, are confined to `outputs/reproducibility_check_v1/`. The requested report path was checked before writing and did not exist.

## Scope and execution safety

The six directory names in the prompt were inspected. The model-input uncertainty directory `outputs/pisa_model_input_uncertainty_v1/` is absent (no manifest or outputs), so the corresponding manifest cannot be audited. `outputs/pisa2025_forecast_sensitivity_v7/run_manifest.json` describes a sensitivity-analysis run, not the stated `scripts/run_pisa_2025_retrospective_tabular.py` benchmark. Its script path is omitted, and its predictions/metrics inputs are in the TimesFM retrospective directory. This is a material manifest-to-prompt/output-path mismatch.

The v1 tabular manifest has no `script`, `script_sha256`, or output hashes. Its named prediction and metric CSVs exist, but the recorded manifest does not permit output hash verification. The v2, TimesFM v2, sensitivity v7, and link-error manifests were read as found. The v2 benchmark and v1 model-input scripts hard-code their original output directories and abort if those paths exist; the retrospective tabular script also hard-codes its output directory; rerunning those scripts would require changing code or runtime behavior. No script was modified. The pairwise link-error audit supports `--output-dir`, so it was rerun safely under the requested check directory. Although this script uses no TimesFM model, the TimesFM-dependent historical arm was not rerun because the prompt explicitly excludes TimesFM-required runs.

The scripts have no documented age/verification date. Direct SHA-256 checks found `scripts/run_pisa_quadratic_trend_benchmark.py` matches the v2 manifest (`3a85cae8…`); `scripts/audit_pisa2025_pairwise_link_errors.py` matches its manifest (`2b1b7bdc…`). The v1 tabular manifest provides no script hash, and sensitivity v7 provides a script hash without identifying its script path. The named v1 model-input script is not the v2 script found in the repository; the v2 script targets the absent v1 output directory.

## Run-by-run results

### Tabular benchmark v1 — `outputs/pisa_tabular_benchmark_v1/`

**MISMATCH [incomplete provenance].** Manifest records panel SHA-256 `91f44322fe036f9aa44dfc5a39c3387495879ff2e381a3a823374e4f2a3fc7f9` and fold manifest SHA-256 `98b343b2b4ba3a15cd4f4f996ae04d41c8b38c9cb8d247c0367a03a1ae4f8e81`; the manifest itself does not record script identity or output hashes. It names `historical_predictions.csv` and `historical_metrics.csv`. A complete output comparison or reproducibility claim is therefore unavailable. Script rerun is infeasible without changing the hard-coded `OUT_DIR` or running modified code, both disallowed by the request.

### Tabular benchmark v2 — `outputs/pisa_tabular_benchmark_v2/`

**INFEASIBLE [safe rerun unavailable without changing the script].** The manifest’s script hash matches the current script (`3a85cae8ed37673b868f6eb6be18fb41ac3573009e707d1a54b1451d1ceb407d`). Recorded panel hash is `91f44322fe036f9aa44dfc5a39c3387495879ff2e381a3a823374e4f2a3fc7f9`; fold-manifest hash is `98b343b2b4ba3a15cd4f4f996ae04d41c8b38c9cb8d247c0367a03a1ae4f8e81`; baseline prediction hash is `6b14edcd60d2d351fdcefaad586892eb98f30f694377cf184a85dd6015b732da`; baseline manifest hash is `204775db54b4796a30aae8fc1bf3cf2d8d11b925eb365288371b7689ba72ad57`. Manifest output hashes are recorded for predictions (`bc6895b14a06e53c2f9e4ceb16d02f019e289b5463ce98a01a43f361fdd92a98`), metrics (`a68ce30c30cf2e020548cda6244ad89e7bacfb7a6570b84ff7901194735d817f`), and diagnostics (`c1f9028c9356fa674e83c94c6644574687e4dc6196ae974775fa276e399d8cd5`). The current script hard-codes the existing output directory and refuses overwrite; a fresh destination cannot be supplied without code/runtime alteration. No comparison of prediction cells was made.

### TimesFM historical zero-shot v2 — `outputs/pisa_timesfm_historical_zeroshot_v2/`

**INFEASIBLE [explicitly excluded TimesFM arm].** Manifest states checkpoint `google/timesfm-3.0-pytorch`, cached revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, distribution 3.0.2, and CPU device; the user directed exclusion of TimesFM-required runs. Panel hash is recorded as `91f44322fe036f9aa44dfc5a39c3387495879ff2e381a3a823374e4f2a3fc7f9`, fold hash as `98b343b2b4ba3a15cd4f4f996ae04d41c8b38c9cb8d247c0367a03a1ae4f8e81`, and tabular input hash as `6b14edcd60d2d351fdcefaad586892eb98f30f694377cf184a85dd6015b732da`. Its manifest lists five outputs but records no output hashes and no script hash. This means exact CSV comparison is not supported by this manifest even aside from the excluded rerun.

### PISA 2025 retrospective tabular / sensitivity v7 — manifest path mismatch

**MISMATCH [manifest does not describe the requested run].** The prompt pairs `scripts/run_pisa_2025_retrospective_tabular.py` with `outputs/pisa2025_forecast_sensitivity_v7/`. The latter manifest instead describes an outcome-exposed sensitivity analysis. Its input predictions hash is `6d12afc9a3cd4ab511315a5748bf781073644790feb74ab3a01144be66b7e25a`; metrics input hash is `773360436b41375f34dea8fd27456405558d6dcb944910fd21c6027c72bfc789`; quality-flag hash is `8be34a6a05655c3d43de98f6f2c5902de06679ef1275364868bcbe1395ea557c`; manifest records script hash `1dfbb5a71ce96aac2d8e0a2324ead8375c9a791cd97a92ca4f97146225d948b2` but no script path. The manifest lists `subset_and_context_metrics.csv` and `paired_country_bootstrap.csv`, but no output hashes. It is not evidence for the requested tabular retrospective script run. That script targets `outputs/pisa_2025_retrospective_tabular_v1/`, not v7, and has no output-directory option. It was not run because doing so safely to the prescribed destination would require altering the script/runtime.

### Model-input uncertainty v1 — `outputs/pisa_model_input_uncertainty_v1/`

**INFEASIBLE [missing run artifacts].** Directory and manifest are absent; no listed input, script, or output hashes could be checked. `scripts/run_pisa_model_input_uncertainty_v2.py` exists and targets this absent v1 directory, but is not the named `_v1.py` script and hard-codes the output path. It was not run because this would require redirecting execution by changing code/runtime and would create a new, non-manifested run rather than reproduce an auditable existing one.

### Pairwise link-error audit v1 — `outputs/pisa2025_pairwise_link_error_v1/`

**REPRODUCED.** The named script hash matches the manifest: `2b1b7bdca838f9aad0d4e97262a95186264a1d4e9a38f836730c36a7d0c6b703`. Source workbook hash recorded by the manifest is `14e958538fc68dc4bf25b47ad11c4f3f90035873aa96616925e5cea0af242fb8`. The script was rerun with `--output-dir outputs/reproducibility_check_v1/pisa2025_pairwise_link_error`; it completed with 1,280 detail rows and 21 summary rows. Both output CSV hashes exactly match the manifest:

| Output | Manifest SHA-256 | Rerun SHA-256 | Result |
|---|---|---|---|
| `country_pairwise_link_error_reconstruction.csv` | `fb54104d0e2af8ab8f0b79f6f0a734fdcfc32bb11289306a47a258075fd01aa8` | `fb54104d0e2af8ab8f0b79f6f0a734fdcfc32bb11289306a47a258075fd01aa8` | exact match |
| `pairwise_link_error_summary.csv` | `73bafa34ff0b89c0e07879a4bfb82dff21eee4009ed282821ebe0d24615c90d2` | `73bafa34ff0b89c0e07879a4bfb82dff21eee4009ed282821ebe0d24615c90d2` | exact match |

The rerun also reports all 21 domain-cycle pairs reconstructed, country invariance within table, and agreement with the NCES reference to 3 decimal places. The regenerated manifest hash differs because its run timestamp/output-directory metadata differs; the two data CSVs are byte-identical. No cell-level discrepancy exists in those CSVs, so no differing column/row is applicable.

## Hash and comparison limitations

For the first five run groups, full live input hash checks and output byte-hash comparisons were not completed as an independent rerun. The supplied manifests provide the recorded hashes stated above, but their available metadata is incomplete in several cases. No claim of CSV-cell agreement is made for those groups. The exact byte match for the pairwise link-error CSVs is the only completed rerun comparison; because those files are byte-identical, their columns, rows, and values match exactly.
