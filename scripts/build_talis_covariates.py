#!/usr/bin/env python3
"""
TALIS 2008 / 2013 / 2018 / 2024 öğretmen kovaryatları

Her ülke için TCHWGT ile ağırlıklı ortalama ve JRR standart hata üretir.
Çıktı: outputs/stage5/talis_covariate_estimates.csv

Sütunlar:
  program, cycle, country_iso3, variable, construct,
  mean, se, n, n_replicates, weight_used, source, lag_years, target_ilsa

Metodolojik notlar:
  - Varyans: JRR (Jackknife Repeated Replication).
    V_JRR = Σ_{r=1}^{R} (θ_r − θ_0)^2
    TRWGT1–TRWGT{R} replicate ağırlıkları metadata'dan otomatik tespit edilir.
    2013/2018: R=100; 2008/2024: mevcut replicate sütun sayısına göre değişir.
    Replicate bulunamazsa se=NaN, n_replicates=0 olarak kaydedilir.
  - Composite scale değişkenleri IDB tarafından zaten ölçeklenmiş WLE/scale scores.
    Yeniden standardize edilmiyor — ham ağırlıklı ortalama kullanılıyor.
  - 2008: tek composite scale (SELFEF — genel öğretmen öz-yeterlilik indeksi);
    2013/2018/2024 gibi 3 ayrı boyuta ayrılmamış. Karşılaştırılabilirlik notu
    gerektirir. IDCNTRY nümerik kod → TALIS2008_IDCNTRY_ISO3 haritası ile ISO3'e çevrilir.
  - 2024: tek birleşik dosya (ttgintt4.sav); T4SECLS / T4SEINS / T4SEENG değişkenleri
    2013/2018 ile canonical olarak uyumlu.
  - Alt-ulusal birimler (AAD, ABA, BFL, CAB, CSH, ENG, INT) filtreleniyor.
  - lag_years: TALIS cycle → en yakın sonraki PISA/TIMSS cycle arasındaki yıl farkı.
"""
from __future__ import annotations

import logging
import pathlib
import glob

import numpy as np
import pandas as pd
import pyreadstat

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
TALIS_BASE   = pathlib.Path("/Users/mrved/Desktop/ILSA Datasets for LLM Extraction/TALIS Datasets")
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"
OUT_CSV      = OUT_DIR / "talis_covariate_estimates.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Alt-ulusal / non-standard TALIS kodları ───────────────────────────────────
SUBNATIONAL = {"AAD", "ABA", "BFL", "CAB", "CSH", "ENG", "INT"}

# ── TALIS 2008: IDCNTRY nümerik kod → ISO3 haritası ─────────────────────────
# Kaynak: TALIS 2008 IDB değer etiketleri (BTGINTT1.sav → IDCNTRY value labels)
TALIS2008_IDCNTRY_ISO3: dict[int, str] = {
    36:  "AUS", 40:  "AUT", 76:  "BRA", 100: "BGR", 208: "DNK",
    233: "EST", 246: "FIN", 250: "FRA", 348: "HUN", 372: "IRL",
    380: "ITA", 392: "JPN", 410: "KOR", 440: "LTU", 484: "MEX",
    528: "NLD", 554: "NZL", 578: "NOR", 616: "POL", 620: "PRT",
    703: "SVK", 705: "SVN", 724: "ESP", 752: "SWE", 756: "CHE",
    792: "TUR", 826: "GBR",
}

# ── Construct harmonization: raw variable → canonical adı ────────────────────
# 2013 ve 2018 için aynı canonical ismi kullanıyoruz; karşılaştırılabilirlik sağlanmış.
CONSTRUCT_MAP = {
    # 2008 → canonical (tek genel indeks; 3 boyuta ayrılmamış)
    "SELFEF":   ("self_efficacy_general",      "Self-Efficacy: General Index (2008 only)"),
    # 2013 → canonical
    "SECLSS":   ("self_efficacy_cls_mgmt",     "Self-Efficacy: Classroom Management"),
    "SEINSS":   ("self_efficacy_instruction",  "Self-Efficacy: Instruction"),
    "SEENGS":   ("self_efficacy_engagement",   "Self-Efficacy: Student Engagement"),
    "TCDISCS":  ("disciplinary_climate",       "Disciplinary Climate"),
    "TCCOLLS":  ("prof_collaboration",         "Professional Collaboration"),
    "TJSENVS":  ("job_sat_environment",        "Job Satisfaction: Work Environment"),
    "TJSPROS":  ("job_sat_profession",         "Job Satisfaction: Profession"),
    "TEFFPROS": ("effective_pd",               "Effective Professional Development"),
    # 2018 → canonical (aynı adlar)
    "T3SECLS":  ("self_efficacy_cls_mgmt",     "Self-Efficacy: Classroom Management"),
    "T3SEINS":  ("self_efficacy_instruction",  "Self-Efficacy: Instruction"),
    "T3SEENG":  ("self_efficacy_engagement",   "Self-Efficacy: Student Engagement"),
    "T3DISC":   ("disciplinary_climate",       "Disciplinary Climate"),
    "T3COLES":  ("prof_collaboration",         "Professional Collaboration"),
    "T3JSENV":  ("job_sat_environment",        "Job Satisfaction: Work Environment"),
    "T3JSPRO":  ("job_sat_profession",         "Job Satisfaction: Profession"),
    "T3EFFPD":  ("effective_pd",               "Effective Professional Development"),
    # 2024 → canonical (aynı adlar; T4 prefix)
    "T4SECLS":  ("self_efficacy_cls_mgmt",     "Self-Efficacy: Classroom Management"),
    "T4SEINS":  ("self_efficacy_instruction",  "Self-Efficacy: Instruction"),
    "T4SEENG":  ("self_efficacy_engagement",   "Self-Efficacy: Student Engagement"),
}

# ── Cycle spesifikasyonları ───────────────────────────────────────────────────
CYCLE_SPECS = [
    {
        "cycle":      2008,
        # Tek birleşik dosya: BTGINTT1.sav (tüm ülkeler)
        "data_dir":   TALIS_BASE / "TALIS 2008 Data" / "SPSS combined",
        "btg_prefix": "BTG",
        "cntry_col":  "IDCNTRY",       # nümerik → TALIS2008_IDCNTRY_ISO3 ile çevrilir
        "weight_col": "TCHWGT",
        "variables":  ["SELFEF"],
        "temporal_targets": {
            "PISA":  {"next_cycle": 2009, "lag_years": 1},
            "TIMSS": {"next_cycle": 2011, "lag_years": 3},
        },
    },
    {
        "cycle":      2013,
        "data_dir":   TALIS_BASE / "TALIS 2013 Data",
        "btg_prefix": "BTG",
        "cntry_col":  "CNTRY",         # ISO3 doğrudan
        "weight_col": "TCHWGT",
        "variables":  ["SECLSS", "SEINSS", "SEENGS", "TCDISCS",
                       "TCCOLLS", "TJSENVS", "TJSPROS", "TEFFPROS"],
        # TALIS 2013 → sonraki ILSA döngüsü (yıl farkı)
        "temporal_targets": {
            "PISA":  {"next_cycle": 2015, "lag_years": 2},
            "TIMSS": {"next_cycle": 2015, "lag_years": 2},
        },
    },
    {
        "cycle":      2018,
        "data_dir":   TALIS_BASE / "TALIS 2018 Data",
        "btg_prefix": "BTG",
        "cntry_col":  "CNTRY",         # ISO3 doğrudan
        "weight_col": "TCHWGT",
        "variables":  ["T3SECLS", "T3SEINS", "T3SEENG", "T3DISC",
                       "T3COLES", "T3JSENV", "T3JSPRO", "T3EFFPD"],
        "temporal_targets": {
            "PISA":  {"next_cycle": 2022, "lag_years": 4},
            "TIMSS": {"next_cycle": 2019, "lag_years": 1},
        },
    },
    {
        "cycle":      2024,
        # Tek birleşik dosya: ttgintt4.sav (TT prefix; tüm ülkeler)
        "data_dir":   TALIS_BASE / "TALIS 2024 Data" / "TALIS2024_teachers_NoESE_SPSS",
        "btg_prefix": "TT",
        "cntry_col":  "CNTRY",         # ISO3 doğrudan
        "weight_col": "TCHWGT",
        "variables":  ["T4SECLS", "T4SEINS", "T4SEENG"],
        "temporal_targets": {
            "PISA":  {"next_cycle": 2025, "lag_years": 1},
        },
    },
]


def weighted_mean(vals: np.ndarray, wts: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    v, w = vals[mask], wts[mask]
    if len(v) == 0:
        return np.nan, 0
    return float(np.average(v, weights=w)), int(len(v))


def _variance_jrr(
    vals: np.ndarray,
    base_wts: np.ndarray,
    rep_wts: list[np.ndarray],
    theta0: float,
) -> float:
    """JRR variance: V = Σ_{r=1}^{R} (θ_r - θ_0)^2."""
    sq = 0.0
    for rw in rep_wts:
        mask = np.isfinite(vals) & np.isfinite(rw) & (rw > 0)
        if mask.sum() < 5:
            continue
        theta_r, _ = weighted_mean(vals[mask], rw[mask])
        if np.isfinite(theta_r):
            sq += (theta_r - theta0) ** 2
    return sq


def _resolve_iso3(spec: dict, df: pd.DataFrame, fpath: str) -> list[str] | None:
    """
    Dosya başına ISO3 kodu döndürür.
    - cntry_col=CNTRY  → her satırda ISO3 string; satır başına ISO3 listesi döner
    - cntry_col=IDCNTRY → nümerik; TALIS2008_IDCNTRY_ISO3 haritası kullanılır;
      tek dosya tüm ülkeleri barındırır → None döner (satır başına groupby yapılmalı)
    """
    col = spec["cntry_col"].upper()
    if col not in df.columns:
        return None
    if col == "IDCNTRY":
        return None  # groupby ile işlenecek
    return df[col].astype(str).str.upper().tolist()


def process_cycle(spec: dict) -> list[dict]:
    d = spec["data_dir"]
    all_sav = glob.glob(str(d / "**/*.sav"), recursive=True)
    btg_files = sorted(
        p for p in all_sav
        if pathlib.Path(p).name.upper().startswith(spec["btg_prefix"].upper())
    )
    if not btg_files:
        log.warning("TALIS %d: %s dosyası bulunamadı", spec["cycle"], spec["btg_prefix"])
        return []

    log.info("TALIS %d: %d dosya bulundu", spec["cycle"], len(btg_files))
    records = []

    for fpath in btg_files:
        # Alt-ulusal filtresi — çok-dosya yapılarında dosya adından ISO3 alınır;
        # tek-dosya yapılarında (2008 BTGINTT1, 2024 ttgintt4) dosya adı ülke kodu taşımaz,
        # filtre groupby döngüsünde satır bazında uygulanır.
        use_numeric_map = (spec["cntry_col"].upper() == "IDCNTRY")
        multi_file = (len(btg_files) > 1)
        if multi_file:
            stem = pathlib.Path(fpath).stem.upper()
            stem_iso3 = stem[3:6] if len(stem) >= 6 else ""
            if stem_iso3 in SUBNATIONAL:
                log.debug("  %s atlandı (alt-ulusal)", stem_iso3)
                continue

        try:
            _, meta = pyreadstat.read_sav(fpath, metadataonly=True)
        except Exception as e:
            log.debug("  %s metadata hatası: %s", pathlib.Path(fpath).name, e)
            continue

        cols_upper = {c.upper(): c for c in meta.column_names}
        avail_vars = [v for v in spec["variables"] if v.upper() in cols_upper]
        wgt_orig   = cols_upper.get(spec["weight_col"].upper())
        cntry_orig = cols_upper.get(spec["cntry_col"].upper())

        if not avail_vars:
            log.debug("  %s: hiç hedef değişken yok", pathlib.Path(fpath).name)
            continue

        # Replicate weight sütunlarını bul (TRWGT1 .. TRWGTn)
        rep_orig = sorted(
            [orig for up, orig in cols_upper.items() if up.startswith("TRWGT") and up[5:].isdigit()],
            key=lambda c: int(c.upper()[5:]),
        )

        load_cols = [cols_upper[v.upper()] for v in avail_vars]
        if wgt_orig:
            load_cols.append(wgt_orig)
        if cntry_orig and cntry_orig not in load_cols:
            load_cols.append(cntry_orig)
        for r in rep_orig:
            if r not in load_cols:
                load_cols.append(r)

        try:
            df, _ = pyreadstat.read_sav(fpath, usecols=load_cols, apply_value_formats=False)
        except Exception as e:
            log.debug("  %s yüklenemedi: %s", pathlib.Path(fpath).name, e)
            continue

        df.columns = [c.upper() for c in df.columns]
        rep_cols_up = [r.upper() for r in rep_orig]

        cntry_col_up = spec["cntry_col"].upper()

        if use_numeric_map:
            # 2008: tek dosya, tüm ülkeler; IDCNTRY nümerik → groupby
            groups = df.groupby(cntry_col_up)
        else:
            # Çok-dosya veya tek-dosya, CNTRY ISO3 string
            iso3_col = df[cntry_col_up].astype(str).str.upper()
            df = df.copy()
            df["_ISO3"] = iso3_col
            groups = df.groupby("_ISO3")

        for group_key, grp in groups:
            if use_numeric_map:
                iso3 = TALIS2008_IDCNTRY_ISO3.get(int(group_key))
                if iso3 is None:
                    log.debug("  IDCNTRY=%s için ISO3 haritası yok", group_key)
                    continue
            else:
                iso3 = str(group_key).upper()

            if iso3 in SUBNATIONAL:
                continue

            if wgt_orig:
                weights = grp[spec["weight_col"].upper()].values.astype(float)
                weight_used = spec["weight_col"]
            else:
                weights = np.ones(len(grp))
                weight_used = "equal"

            # Replicate ağırlıklarını grup için hazırla
            grp_rep_wts = [
                grp[rc].values.astype(float)
                for rc in rep_cols_up if rc in grp.columns
            ]

            for var in avail_vars:
                canonical, construct_label = CONSTRUCT_MAP.get(var.upper(), (var.upper(), var.upper()))
                vals = grp[var.upper()].values.astype(float)
                mu, n = weighted_mean(vals, weights)
                if not np.isfinite(mu) or n < 30:
                    continue

                # JRR standart hata
                if grp_rep_wts:
                    V = _variance_jrr(vals, weights, grp_rep_wts, mu)
                    se = float(np.sqrt(max(V, 0.0)))
                    n_rep = len(grp_rep_wts)
                else:
                    se = float("nan")
                    n_rep = 0

                for ilsa_prog, tgt in spec["temporal_targets"].items():
                    records.append({
                        "program":         "TALIS",
                        "cycle":           spec["cycle"],
                        "country_iso3":    iso3,
                        "variable":        var.upper(),
                        "construct":       canonical,
                        "construct_label": construct_label,
                        "mean":            round(mu, 4),
                        "se":              round(se, 6) if np.isfinite(se) else None,
                        "n":               n,
                        "n_replicates":    n_rep,
                        "weight_used":     weight_used,
                        "source":          pathlib.Path(fpath).name,
                        "target_program":  ilsa_prog,
                        "target_cycle":    tgt["next_cycle"],
                        "lag_years":       tgt["lag_years"],
                    })

    return records


def main():
    all_records = []
    for spec in CYCLE_SPECS:
        recs = process_cycle(spec)
        all_records.extend(recs)
        log.info("  → %d kayıt (cycle=%d)", len(recs), spec["cycle"])

    if not all_records:
        log.error("Hiç kayıt üretilemedi.")
        return

    df = pd.DataFrame(all_records)

    # Duplicate kontrolü: (country_iso3, cycle, variable, target_program) benzersiz olmalı
    dup_key = ["country_iso3", "cycle", "variable", "target_program"]
    n_dup = df.duplicated(subset=dup_key).sum()
    if n_dup:
        log.error("DUPLICATE HATASI: %d yinelenen (country, cycle, variable, target) satırı!", n_dup)
    else:
        log.info("Duplicate kontrolü: 0 yineleme ✓")

    df.to_csv(OUT_CSV, index=False)
    log.info("Kaydedildi: %s (%d satır)", OUT_CSV, len(df))

    # Özet: cycle × construct × ülke sayısı
    log.info("\nÖzet (cycle × construct, ülke sayısı):")
    summary = (
        df[df["target_program"] == "PISA"]
        .groupby(["cycle", "construct"])["country_iso3"]
        .nunique()
        .unstack("construct", fill_value=0)
    )
    log.info("\n%s", summary.to_string())

    # Cycle availability tablosu
    print("\n── Availability: cycle × ülke (PISA target) ──")
    avail = (
        df[df["target_program"] == "PISA"]
        .groupby(["cycle", "construct"])["country_iso3"]
        .nunique()
    )
    print(avail.to_string())

    print("\n── Örnek satırlar ──")
    print(df[df["country_iso3"] == "TUR"][
        ["cycle", "country_iso3", "variable", "construct", "mean", "n", "target_program", "lag_years"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
