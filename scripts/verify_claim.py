#!/usr/bin/env python3
"""
ILSA Deterministic Verification Engine — Phase 1

Bir Sheet-5 iddiasını SPSS mikroverisinden deterministik olarak doğrular.
Tüm istatistikler bu scriptten çıkar; LLM hiçbir sayı üretmez.

Psikometrik prensipler:
  - Ağırlıklı korelasyon: tasarım ağırlığı (W_FSTUWT) ile hesaplanır
  - Varyans tahmini: BRR_Fay (PISA) — V = 1/(R·(1-k)²) · Σ(θ̂_r - θ̂_0)²
  - PV pooling: Rubin (1987) — Ū + (1+1/m)·B toplam varyansı
  - CI: Fisher z dönüşümü üzerinden, sonra geri dönüşüm
  - Eksik veri: listwise deletion, n_total ve n_analytic ayrı belgelenir

Referanslar:
  Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.
  OECD (2017). PISA 2015 Technical Report. Chapter 8: BRR_Fay.
  Fay, R.E. (1989). Theory and application of replicate weighting.

Kullanım:
  python scripts/verify_claim.py --claim-id CLM_PISA_2012_000001
  python scripts/verify_claim.py --claim-id CLM_PISA_2012_000001 \\
      --checks correlation_direction correlation_magnitude
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyreadstat

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DEFAULT_XLSX  = PROJECT_ROOT / "outputs" / "ILSA_Meta_Analysis_Dataset_CLEAN.xlsx"
CATALOG_DB    = PROJECT_ROOT / "outputs" / "ilsa_microdata_catalog" / "duckdb" / "ilsa_microdata_catalog.duckdb"
EVIDENCE_ROOT = PROJECT_ROOT / "outputs" / "verification" / "runs"
ENGINE_VERSION = "1.0.0-phase1"

# BRR_Fay sabitleri (PISA Technical Report, Ch.8)
BRR_FAY_K  = 0.5   # Fay katsayısı
BRR_FAY_R  = 80    # Replicate weight sayısı
BRR_FAY_DENOM = BRR_FAY_R * (1 - BRR_FAY_K) ** 2  # = 80 × 0.25 = 20.0

# Yön sınıflandırma eşiği (pratik anlam; korelasyon için)
DIRECTION_THRESHOLD = 0.05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modül 1 — Claim Loader
# ---------------------------------------------------------------------------

def load_claim(xlsx: Path, claim_id: str) -> dict:
    df = pd.read_excel(xlsx, sheet_name="5_Verification_Claims", dtype=str)
    row = df[df["claim_id"] == claim_id]
    if row.empty:
        raise ValueError(f"Sheet 5'te bulunamadı: {claim_id}")
    claim = row.iloc[0].to_dict()
    if str(claim.get("verification_eligible", "")).upper() != "TRUE":
        raise ValueError(f"{claim_id} verification_eligible=FALSE; doğrulama yapılamaz.")
    return claim


# ---------------------------------------------------------------------------
# Modül 2 — Catalog Resolver
# ---------------------------------------------------------------------------

def resolve_catalog(claim: dict) -> dict:
    """
    DuckDB'den dosya yolunu ve değişken varlığını doğrula.
    Önce claim'deki catalog_file_id'yi kullan; bulamazsa
    isa+cycle+file_role ile ara.
    """
    con = duckdb.connect(str(CATALOG_DB), read_only=True)

    file_id = str(claim.get("catalog_file_id", "")).strip()
    if file_id and file_id != "nan" and not file_id.startswith("PISA_pending"):
        row = con.execute(
            "SELECT * FROM catalog_files WHERE file_id=?", [file_id]
        ).df()
    else:
        isa   = str(claim.get("normalized_isa", "")).strip()
        cycle = str(claim.get("normalized_cycle", "")).strip()
        row   = con.execute(
            """SELECT * FROM catalog_files
               WHERE isa=? AND cycle=? AND file_role='student'
               AND file_format IN ('.sav','.SAV')
               ORDER BY n_rows DESC LIMIT 1""",
            [isa, int(float(cycle))]
        ).df()

    if row.empty:
        raise RuntimeError("Katalogda eşleşen dosya bulunamadı.")

    info = row.iloc[0].to_dict()
    file_path = Path(info["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"Katalog kaydı var ama dosya yok: {file_path}")

    # Değişken varlığını doğrula
    outcome_vars = [v.strip() for v in str(claim.get("official_outcome_vars", "")).split(";") if v.strip()]
    predictor_vars = [v.strip() for v in str(claim.get("official_predictor_vars", "")).split(";") if v.strip()]
    all_check_vars = outcome_vars + predictor_vars

    if all_check_vars:
        placeholders = ",".join(["?"] * len(all_check_vars))
        found = con.execute(
            f"SELECT variable_name FROM catalog_variables "
            f"WHERE file_id=? AND variable_name IN ({placeholders})",
            [info["file_id"]] + all_check_vars
        ).df()["variable_name"].tolist()
        missing = [v for v in all_check_vars if v not in found]
        if missing:
            log.warning("Değişkenler katalogda yok (yine de devam): %s", missing)

    rep_wgts = con.execute(
        "SELECT variable_name FROM catalog_variables "
        "WHERE file_id=? AND is_replicate_weight=true ORDER BY variable_name",
        [info["file_id"]]
    ).df()["variable_name"].tolist()

    con.close()
    return {
        "file_id":        info["file_id"],
        "file_path":      str(file_path),
        "weight_method":  info.get("weight_method"),
        "primary_weight": info.get("primary_weight_var"),
        "rep_weights":    rep_wgts,
        "isa":            info.get("isa"),
    }


# ---------------------------------------------------------------------------
# Modül 3 — Data Loader
# ---------------------------------------------------------------------------

def load_data(
    file_path: str,
    country_iso3: str,
    country_var: str,
    predictor: str,
    pv_vars: list[str],
    weight_var: str,
    rep_weight_vars: list[str],
) -> pd.DataFrame:
    """
    Sadece gerekli kolonları yükle, ülke filtresi uygula.
    Eksik veri protokolü: listwise deletion, n_total / n_analytic belgelenir.
    """
    needed = [country_var, predictor] + pv_vars + [weight_var] + rep_weight_vars

    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df, _ = pyreadstat.read_sav(
                file_path, usecols=needed,
                apply_value_formats=False, encoding=enc
            )
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Dosya okunamadı: {file_path}")

    # Ülke filtresi
    country_col = df[country_var].astype(str).str.strip()
    df = df[country_col == country_iso3].copy()
    n_total = len(df)
    if n_total == 0:
        raise ValueError(f"Ülke kodu '{country_iso3}' için hiç satır bulunamadı.")

    # Listwise deletion — yalnızca analitik değişkenler
    analytic_vars = [predictor] + pv_vars + [weight_var]
    n_before = len(df)
    df = df.dropna(subset=analytic_vars)
    df = df[df[weight_var] > 0]
    n_analytic = len(df)

    log.info(
        "Veri yüklendi: %s | n_total=%d | n_eksik_silindi=%d | n_analytic=%d",
        country_iso3, n_total, n_before - n_analytic, n_analytic
    )

    if n_analytic < 100:
        raise ValueError(f"Analitik n={n_analytic} < 100; sonuçlar güvenilmez.")

    return df, {"n_total": n_total, "n_analytic": n_analytic,
                "n_missing_deleted": n_before - n_analytic}


# ---------------------------------------------------------------------------
# Modül 4 — Weight Engine: Ağırlıklı Pearson Korelasyonu
# ---------------------------------------------------------------------------

def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Tasarım-ağırlıklı Pearson korelasyonu.
    w normalize edilmeden kullanılır (PISA Technical Report formülü).
    """
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov  = np.sum(w * (x - mx) * (y - my))
    sx   = np.sqrt(np.sum(w * (x - mx) ** 2))
    sy   = np.sqrt(np.sum(w * (y - my) ** 2))
    if sx == 0 or sy == 0:
        return np.nan
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Modül 5 — BRR_Fay Varyans Tahmini
# ---------------------------------------------------------------------------

def brr_fay_variance(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    primary_wgt: str,
    rep_wgt_cols: list[str],
    theta_full: float,
) -> float:
    """
    BRR Fay (k=0.5) varyans tahmini — PISA Technical Report Ch.8.

    V_BRR_Fay(θ̂) = 1 / (R · (1-k)²) · Σ_{r=1}^{R} (θ̂_r - θ̂_0)²

    θ̂_0 : tam ağırlık (W_FSTUWT) ile hesaplanan tahmin
    θ̂_r : r. replicate ağırlıkla hesaplanan tahmin
    """
    if not rep_wgt_cols:
        return np.nan

    x = df[predictor].values
    y = df[outcome].values

    sq_sum = 0.0
    for col in rep_wgt_cols:
        w_r = df[col].values
        theta_r = weighted_pearson(x, y, w_r)
        sq_sum += (theta_r - theta_full) ** 2

    return sq_sum / BRR_FAY_DENOM


# ---------------------------------------------------------------------------
# Modül 6 — Rubin (1987) Pooling — PV'ler üzerinde
# ---------------------------------------------------------------------------

def rubin_pool(
    df: pd.DataFrame,
    predictor: str,
    pv_vars: list[str],
    primary_wgt: str,
    rep_wgt_cols: list[str],
) -> dict:
    """
    Rubin (1987) kurallarıyla m PV üzerinde havuzlama.

    θ̄  = (1/m) Σ θ̂_j                 nokta tahmini
    Ū  = (1/m) Σ U_j                  içi-impütasyon varyans (BRR_Fay)
    B  = (1/(m-1)) Σ (θ̂_j - θ̄)²     arası-impütasyon varyans
    T  = Ū + (1 + 1/m)·B              toplam varyans (Rubin, 1987)
    SE = √T

    Fisher z dönüşümü ile CI hesabı (korelasyon normal dağılmaz):
    z = arctanh(r), CI_z = z ± 1.96·SE_z, sonra tanh() ile geri dönüşüm.
    """
    m   = len(pv_vars)
    x   = df[predictor].values
    w0  = df[primary_wgt].values

    thetas: list[float] = []
    Us:     list[float] = []

    for pv in pv_vars:
        y       = df[pv].values
        theta_j = weighted_pearson(x, y, w0)
        U_j     = brr_fay_variance(df, predictor, pv, primary_wgt, rep_wgt_cols, theta_j)
        thetas.append(theta_j)
        Us.append(U_j)
        log.debug("  %s: θ̂=%.4f  U=%.6f", pv, theta_j, U_j)

    theta_bar = float(np.mean(thetas))
    U_bar     = float(np.nanmean(Us))
    B         = float(np.var(thetas, ddof=1))
    T         = U_bar + (1 + 1 / m) * B
    SE        = float(np.sqrt(T)) if T >= 0 else np.nan

    # Fisher z — CI
    z_bar    = np.arctanh(np.clip(theta_bar, -0.9999, 0.9999))
    SE_z     = SE / (1 - theta_bar ** 2) if abs(theta_bar) < 1 else np.nan
    ci_lo_z  = z_bar - 1.96 * SE_z
    ci_hi_z  = z_bar + 1.96 * SE_z
    ci_lo    = float(np.tanh(ci_lo_z))
    ci_hi    = float(np.tanh(ci_hi_z))

    # Yön sınıflandırması
    if theta_bar > DIRECTION_THRESHOLD:
        direction = "Positive"
    elif theta_bar < -DIRECTION_THRESHOLD:
        direction = "Negative"
    else:
        direction = "Null"

    return {
        "theta_bar":    round(theta_bar, 6),
        "SE":           round(SE, 6),
        "T":            round(T, 8),
        "U_bar":        round(U_bar, 8),
        "B":            round(B, 8),
        "ci_95_lo":     round(ci_lo, 4),
        "ci_95_hi":     round(ci_hi, 4),
        "direction":    direction,
        "m_pv":         m,
        "per_pv":       [round(t, 6) for t in thetas],
        "per_pv_U":     [round(u, 8) for u in Us],
        "pv_vars_used": pv_vars,
    }


# ---------------------------------------------------------------------------
# Modül 7 — Match Status
# ---------------------------------------------------------------------------

def determine_match_status(
    paper_direction: str,
    replicated_direction: str,
    paper_value: str | None,
    replicated_value: float,
) -> str:
    """
    Match statüsünü belirle.

    Full_Match       : yön eşleşti + büyüklük %10 tolerans içinde
    Direction_Match  : yön eşleşti, büyüklük karşılaştırılmadı/raporlanmadı
    Direction_Mismatch: yönler zıt
    Partial_Match    : biri eşleşti, diğeri belirsiz
    Not_Comparable   : makale yeterli istatistik raporlamamış
    """
    pd_norm = str(paper_direction).strip().lower()
    rd_norm = str(replicated_direction).strip().lower()

    if pd_norm in ("nan", "none", "", "not_reported"):
        return "Not_Comparable"

    dir_match = pd_norm == rd_norm

    # Büyüklük karşılaştırması — makale sayısal değer raporladıysa
    import math
    try:
        pval = float(paper_value)
        if math.isnan(pval):
            raise ValueError
        magnitude_ok = abs(pval - replicated_value) <= max(0.10 * abs(pval), 0.05)
    except (TypeError, ValueError):
        pval = None
        magnitude_ok = None

    if not dir_match:
        return "Direction_Mismatch"
    if magnitude_ok is True:
        return "Full_Match"
    if magnitude_ok is False:
        return "Partial_Match"
    # Makale sadece yön raporlamış, sayısal değer yok → yön eşleşti yeterli
    return "Direction_Match"


# ---------------------------------------------------------------------------
# Modül 8 — Evidence Bundle + Sheet 6 Yazıcı
# ---------------------------------------------------------------------------

def _evidence_hash(file_path: str, claim_id: str, country: str) -> str:
    payload = f"{file_path}::{claim_id}::{country}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_evidence(
    claim_id: str,
    claim: dict,
    catalog_info: dict,
    data_info: dict,
    pool: dict,
    match_status: str,
    run_start: datetime,
    run_end: datetime,
) -> Path:
    ev_dir = EVIDENCE_ROOT / claim_id
    ev_dir.mkdir(parents=True, exist_ok=True)

    # per_pv_results.csv
    per_pv_df = pd.DataFrame({
        "pv_var":     pool["pv_vars_used"],
        "theta":      pool["per_pv"],
        "U_brr_fay":  pool["per_pv_U"],
    })
    per_pv_df.to_csv(ev_dir / "per_pv_results.csv", index=False)

    # summary.json
    summary = {
        "claim_id":            claim_id,
        "engine_version":      ENGINE_VERSION,
        "run_timestamp":       run_start.isoformat(),
        "run_duration_sec":    (run_end - run_start).total_seconds(),
        "file_id":             catalog_info["file_id"],
        "file_path":           catalog_info["file_path"],
        "isa":                 catalog_info["isa"],
        "country_filter":      str(claim.get("countries_iso3", "")),
        "predictor":           str(claim.get("official_predictor_vars", "")),
        "outcome_pv_set":      str(claim.get("official_outcome_vars", "")),
        "weight_var":          str(claim.get("weight_var", "")),
        "weight_method":       catalog_info.get("weight_method"),
        "pv_rule":             "rubin_rules",
        "n_pv":                pool["m_pv"],
        "n_rep_weights":       len(catalog_info.get("rep_weights", [])),
        "brr_fay_k":           BRR_FAY_K,
        "brr_fay_R":           BRR_FAY_R,
        "n_total":             data_info["n_total"],
        "n_analytic":          data_info["n_analytic"],
        "n_missing_deleted":   data_info["n_missing_deleted"],
        "theta_bar":           pool["theta_bar"],
        "SE":                  pool["SE"],
        "T_total_variance":    pool["T"],
        "U_bar_within":        pool["U_bar"],
        "B_between":           pool["B"],
        "ci_95_lo":            pool["ci_95_lo"],
        "ci_95_hi":            pool["ci_95_hi"],
        "replicated_direction": pool["direction"],
        "match_status":        match_status,
        "evidence_hash":       _evidence_hash(
            catalog_info["file_path"], claim_id,
            str(claim.get("countries_iso3", ""))
        ),
    }
    (ev_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # manifest.json — yeniden üretilebilirlik
    manifest = {
        "claim_id":      claim_id,
        "catalog_file":  catalog_info["file_path"],
        "script":        str(Path(__file__).resolve()),
        "engine_version": ENGINE_VERSION,
        "built_at":      run_start.isoformat(),
        "evidence_hash": summary["evidence_hash"],
    }
    (ev_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    log.info("Evidence bundle: %s", ev_dir)
    return ev_dir


def update_sheet6(
    xlsx: Path,
    claim_id: str,
    claim: dict,
    pool: dict,
    data_info: dict,
    catalog_info: dict,
    match_status: str,
    ev_dir: Path,
    run_start: datetime,
    run_end: datetime,
) -> None:
    result_id = f"RES_{claim_id}_correlation_direction_001"

    new_row = {
        "result_id":              result_id,
        "claim_id":               claim_id,
        "check_type":             "correlation_direction",
        "check_subtype":          "weighted_pearson_rubin_pooled_brr_fay",
        "paper_reported_value":   str(claim.get("paper_reported_value", "")),
        "paper_reported_direction": str(claim.get("paper_reported_direction", "")),
        "paper_significance":     str(claim.get("paper_significance", "")),
        "replicated_value":       pool["theta_bar"],
        "replicated_se":          pool["SE"],
        "replicated_p_value":     None,
        "replicated_direction":   pool["direction"],
        "ci_95_lo":               pool["ci_95_lo"],
        "ci_95_hi":               pool["ci_95_hi"],
        "match_delta":            None,
        "match_status":           match_status,
        "alignment_narrative":    (
            f"Rubin (1987) pooled r={pool['theta_bar']:.4f} "
            f"(SE={pool['SE']:.4f}, 95% CI [{pool['ci_95_lo']:.4f}, {pool['ci_95_hi']:.4f}]) "
            f"from {pool['m_pv']} PVs with BRR_Fay variance (R=80, k=0.5). "
            f"n_analytic={data_info['n_analytic']} "
            f"({data_info['n_missing_deleted']} cases deleted listwise)."
        ),
        "n_analytic":             data_info["n_analytic"],
        "n_total":                data_info["n_total"],
        "n_missing_deleted":      data_info["n_missing_deleted"],
        "pv_rule_applied":        "rubin_rules",
        "weight_var_applied":     str(claim.get("weight_var", "")),
        "weight_method_applied":  catalog_info.get("weight_method"),
        "country_filter_applied": str(claim.get("countries_iso3", "")),
        "evidence_path":          str(ev_dir) + "/",
        "evidence_artifact_list": "manifest.json;summary.json;per_pv_results.csv",
        "engine_version":         ENGINE_VERSION,
        "run_timestamp":          run_start.isoformat(),
        "run_duration_sec":       round((run_end - run_start).total_seconds(), 2),
        "failure_code":           None,
    }

    try:
        df6 = pd.read_excel(xlsx, sheet_name="6_Verification_Results")
        df6 = df6[df6["claim_id"] != claim_id]
        df6 = pd.concat([df6, pd.DataFrame([new_row])], ignore_index=True)
    except Exception:
        df6 = pd.DataFrame([new_row])

    with pd.ExcelWriter(xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        df6.to_excel(w, sheet_name="6_Verification_Results", index=False)

    log.info("Sheet 6 güncellendi: %s → %s", claim_id, match_status)


# ---------------------------------------------------------------------------
# Ana orkestrasyon
# ---------------------------------------------------------------------------

def verify_claim(
    claim_id: str,
    *,
    checks: list[str],
    xlsx: Path = DEFAULT_XLSX,
) -> dict:
    run_start = datetime.now(timezone.utc)
    log.info("=== verify_claim %s başlıyor ===", claim_id)

    # 1. Claim yükle
    claim = load_claim(xlsx, claim_id)
    log.info("Claim yüklendi: ISA=%s cycle=%s ülke=%s",
             claim.get("normalized_isa"), claim.get("normalized_cycle"),
             claim.get("countries_iso3"))

    # 2. Katalog çöz
    catalog_info = resolve_catalog(claim)
    log.info("Dosya çözüldü: %s", Path(catalog_info["file_path"]).name)
    log.info("Weight method: %s | Replicate ağırlık sayısı: %d",
             catalog_info["weight_method"], len(catalog_info["rep_weights"]))

    # 3. PV değişkenlerini çıkar
    outcome_key  = str(claim.get("official_outcome_vars", ""))
    predictor    = str(claim.get("official_predictor_vars", "")).split(";")[0].strip()
    weight_var   = str(claim.get("weight_var", "W_FSTUWT")).strip()
    country_iso3 = str(claim.get("countries_iso3", "")).strip()
    country_var  = "CNT" if catalog_info["isa"] == "PISA" else "IDCNTRY"

    # PV setini katalogdan al
    con = duckdb.connect(str(CATALOG_DB), read_only=True)
    pv_df = con.execute(
        """SELECT variable_name, pv_draw_index
           FROM catalog_variables
           WHERE file_id=? AND is_pv=true AND pv_domain='math'
           ORDER BY pv_draw_index""",
        [catalog_info["file_id"]]
    ).df()
    con.close()

    if pv_df.empty:
        raise RuntimeError("Katalogda matematk PV değişkeni bulunamadı.")
    pv_vars = pv_df["variable_name"].tolist()
    log.info("PV seti: %s (%d adet)", pv_vars[0] + "…" + pv_vars[-1], len(pv_vars))

    # 4. Veri yükle
    df, data_info = load_data(
        file_path       = catalog_info["file_path"],
        country_iso3    = country_iso3,
        country_var     = country_var,
        predictor       = predictor,
        pv_vars         = pv_vars,
        weight_var      = weight_var,
        rep_weight_vars = catalog_info["rep_weights"],
    )

    # 5. Rubin pooling
    log.info("Rubin pooling başlıyor (%d PV × %d replicate weight) …",
             len(pv_vars), len(catalog_info["rep_weights"]))
    pool = rubin_pool(
        df            = df,
        predictor     = predictor,
        pv_vars       = pv_vars,
        primary_wgt   = weight_var,
        rep_wgt_cols  = catalog_info["rep_weights"],
    )
    log.info(
        "Sonuç: θ̄=%.4f  SE=%.4f  95%%CI [%.4f, %.4f]  yön=%s",
        pool["theta_bar"], pool["SE"], pool["ci_95_lo"], pool["ci_95_hi"], pool["direction"]
    )
    log.info("Rubin bileşenleri: Ū=%.6f  B=%.6f  T=%.6f",
             pool["U_bar"], pool["B"], pool["T"])

    # 6. Match status
    match_status = determine_match_status(
        paper_direction    = str(claim.get("paper_reported_direction", "")),
        replicated_direction = pool["direction"],
        paper_value        = str(claim.get("paper_reported_value", "")),
        replicated_value   = pool["theta_bar"],
    )
    log.info("Match status: %s", match_status)

    run_end = datetime.now(timezone.utc)

    # 7. Evidence bundle yaz
    ev_dir = write_evidence(
        claim_id, claim, catalog_info, data_info,
        pool, match_status, run_start, run_end,
    )

    # 8. Sheet 6 güncelle
    update_sheet6(
        xlsx, claim_id, claim, pool, data_info,
        catalog_info, match_status, ev_dir, run_start, run_end,
    )

    result = {
        "claim_id":              claim_id,
        "engine_version":        ENGINE_VERSION,
        "isa":                   catalog_info["isa"],
        "country":               country_iso3,
        "predictor":             predictor,
        "pv_set":                f"{pv_vars[0]}–{pv_vars[-1]} ({len(pv_vars)} PV)",
        "n_analytic":            data_info["n_analytic"],
        "n_missing_deleted":     data_info["n_missing_deleted"],
        "replicated_value":      pool["theta_bar"],
        "replicated_se":         pool["SE"],
        "ci_95":                 [pool["ci_95_lo"], pool["ci_95_hi"]],
        "rubin_U_bar":           pool["U_bar"],
        "rubin_B":               pool["B"],
        "rubin_T":               pool["T"],
        "brr_fay_R":             BRR_FAY_R,
        "brr_fay_k":             BRR_FAY_K,
        "replicated_direction":  pool["direction"],
        "paper_direction":       str(claim.get("paper_reported_direction", "")),
        "match_status":          match_status,
        "pv_rule_applied":       "rubin_rules",
        "evidence_path":         str(ev_dir),
        "run_duration_sec":      round((run_end - run_start).total_seconds(), 2),
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument(
        "--checks", nargs="+",
        default=["correlation_direction"],
        choices=[
            "correlation_direction", "correlation_magnitude",
            "descriptive", "weighted_mean",
            "regression_coefficient_sign", "sample_size",
        ],
    )
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    args = parser.parse_args()

    result = verify_claim(
        args.claim_id,
        checks=args.checks,
        xlsx=args.xlsx.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result.get("match_status") in ("Full_Match", "Direction_Match") else 1)


if __name__ == "__main__":
    main()
