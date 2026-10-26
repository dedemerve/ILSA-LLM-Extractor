#!/usr/bin/env python3
"""
Zero-shot TimesFM 2.5 LOCO evaluation across non-PISA ILSA programs.

PISA sonuçları run_pisa_timesfm25_ils_adapter.py'dan geldiği için bu script
yalnızca TIMSS_G8, TIMSS_G4, PIRLS, ICCS ve ICILS programlarını değerlendirir.

Değerlendirme tasarımı:
  - Sıfır-atış çıkarımı (LoRA yok; PISA LoRA PISA+TIMSS+PIRLS üzerinde eğitildiği
    için TIMSS/PIRLS test setlerine uygulanması dağılım içi olurdu)
  - LOCO: her test döngüsü için bağlam = aynı program/alan/ülkenin önceki döngüleri
  - Minimum bağlam: MIN_CONTEXT = 2 önceki gözlem
  - Model: TimesFM 2.5 (google/timesfm-2.5-200m-transformers, local_files_only=True)

Çıktı: outputs/ilsa_timesfm_all_programs_v1/
  predictions.csv         — her ülke/döngü/model için tahmin ve hata
  fold_metrics.csv        — her test döngüsü ve alan için MAE/RMSE
  equal_fold_summary.csv  — PISA equal_fold_summary.csv ile aynı şema
  run_manifest.json
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TimesFm2_5ModelForPrediction

ROOT         = Path(__file__).resolve().parents[1]
UNIFIED      = ROOT / "outputs/stage5/unified_panel.csv"
OUT_DIR      = ROOT / "outputs/ilsa_timesfm_all_programs_v1"
MODEL_ID     = "google/timesfm-2.5-200m-transformers"

MIN_CONTEXT          = 2
FORECAST_CONTEXT_LEN = 32
BATCH_SIZE           = 8
DEVICE               = "cpu"

# program label in unified_panel → (eval_program_label, [(domain, score_col), ...])
PROGRAM_CONFIGS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "TIMSS": (
        "TIMSS_G8",
        [("mathematics", "TIMSS_mathematics"), ("science", "TIMSS_science")],
    ),
    "TIMSS_G4": (
        "TIMSS_G4",
        [("mathematics", "TIMSS_G4_mathematics"), ("science", "TIMSS_G4_science")],
    ),
    "PIRLS": (
        "PIRLS",
        [("reading", "PIRLS_reading")],
    ),
    "ICCS": (
        "ICCS",
        [("civic_knowledge", "ICCS_civic_knowledge")],
    ),
    "ICILS": (
        "ICILS",
        [("computer_literacy", "ICILS_computer_literacy")],
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model() -> TimesFm2_5ModelForPrediction:
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    return model.to(DEVICE).eval()


def _batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.no_grad()
def predict_zero_shot(
    model: TimesFm2_5ModelForPrediction,
    contexts: list[np.ndarray],
) -> np.ndarray:
    """Run one-step-ahead zero-shot forecast for a list of context arrays."""
    results: list[float] = []
    for batch in _batches(contexts, BATCH_SIZE):
        past = [
            torch.as_tensor(ctx, dtype=torch.float32, device=DEVICE)
            for ctx in batch
        ]
        output = model(
            past_values=past,
            forecast_context_len=FORECAST_CONTEXT_LEN,
            truncate_negative=False,
        )
        results.extend(output.mean_predictions[:, 0].detach().cpu().numpy().tolist())
    values = np.asarray(results, dtype=float)
    if len(values) != len(contexts) or not np.isfinite(values).all():
        raise ValueError(
            f"Forecast mismatch or non-finite value: expected {len(contexts)}, "
            f"got {len(values)}, finite={np.isfinite(values).all()}"
        )
    return values


def run_loco_program(
    panel: pd.DataFrame,
    panel_program: str,
    eval_program: str,
    domain: str,
    score_col: str,
    model: TimesFm2_5ModelForPrediction,
) -> list[dict]:
    """LOCO evaluation for one program/domain.

    For each test cycle, uses all available prior cycles of the same
    program/domain/country as context. Countries must have ≥ MIN_CONTEXT
    prior observations to be included in the test fold.
    """
    subset = panel[
        (panel["program"] == panel_program)
        & (panel["domain"] == domain)
        & panel[score_col].notna()
    ][["country_iso3", "cycle", score_col]].copy()
    subset["cycle"] = subset["cycle"].astype(int)

    cycles = sorted(subset["cycle"].unique())
    rows: list[dict] = []

    for test_cycle in cycles:
        prior_cycles = [c for c in cycles if c < test_cycle]
        if len(prior_cycles) < MIN_CONTEXT:
            continue

        test_countries = subset[subset["cycle"] == test_cycle]["country_iso3"].tolist()
        examples: list[dict] = []
        for country in test_countries:
            hist = subset[
                (subset["country_iso3"] == country) & (subset["cycle"] < test_cycle)
            ].sort_values("cycle")
            if len(hist) < MIN_CONTEXT:
                continue
            true_val = subset[
                (subset["country_iso3"] == country) & (subset["cycle"] == test_cycle)
            ][score_col].iloc[0]
            if not np.isfinite(true_val):
                continue
            examples.append({
                "country_iso3": country,
                "n_context":    len(hist),
                "last_cycle":   int(hist["cycle"].iloc[-1]),
                "true_score":   float(true_val),
                "context":      hist[score_col].astype(float).to_numpy(dtype=np.float32),
            })

        if not examples:
            print(
                f"  {eval_program}/{domain} test_cycle={test_cycle}: "
                "no eligible countries, skipping"
            )
            continue

        contexts = [ex["context"] for ex in examples]
        preds = predict_zero_shot(model, contexts)

        for ex, pred in zip(examples, preds):
            err = float(pred) - ex["true_score"]
            rows.append({
                "program":      eval_program,
                "domain":       domain,
                "test_cycle":   test_cycle,
                "country_iso3": ex["country_iso3"],
                "n_context":    ex["n_context"],
                "last_cycle":   ex["last_cycle"],
                "model":        "TimesFM2_5_zero_shot",
                "true_score":   ex["true_score"],
                "prediction":   float(pred),
                "signed_error": err,
                "absolute_error": abs(err),
                "squared_error":  err ** 2,
            })

        n = len(examples)
        mae = float(np.mean([abs(r["signed_error"]) for r in rows[-n:]]))
        print(
            f"  {eval_program}/{domain} test_cycle={test_cycle}: "
            f"n={n}, MAE={mae:.2f}"
        )

    return rows


def build_fold_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (prog, dom, cyc, mdl), g in pred_df.groupby(
        ["program", "domain", "test_cycle", "model"], sort=True
    ):
        err = g["signed_error"].to_numpy(float)
        rows.append({
            "program":    prog,
            "domain":     dom,
            "test_cycle": int(cyc),
            "model":      mdl,
            "n":          len(g),
            "MAE":        float(np.abs(err).mean()),
            "RMSE":       float(np.sqrt(np.square(err).mean())),
            "mean_signed_error": float(err.mean()),
        })
    return pd.DataFrame(rows)


def build_equal_fold_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_metrics.groupby(["program", "domain", "model"], as_index=False)
        .agg(
            equal_fold_MAE=("MAE",   "mean"),
            equal_fold_RMSE=("RMSE", "mean"),
            equal_fold_mean_signed_error=("mean_signed_error", "mean"),
            folds=("test_cycle", "nunique"),
        )
    )


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT_DIR}")
    if not UNIFIED.is_file():
        raise FileNotFoundError(UNIFIED)

    unified_hash = sha256(UNIFIED)
    panel = pd.read_csv(UNIFIED, dtype={"country_iso3": str})

    print(f"Unified panel: {len(panel)} rows, hash={unified_hash[:12]}...", flush=True)
    print(f"Loading TimesFM 2.5 ({MODEL_ID})...", flush=True)
    model = load_model()

    all_preds: list[dict] = []

    for panel_prog, (eval_prog, domain_configs) in PROGRAM_CONFIGS.items():
        for domain, score_col in domain_configs:
            n_countries = panel[
                (panel["program"] == panel_prog)
                & (panel["domain"] == domain)
                & panel[score_col].notna()
            ]["country_iso3"].nunique()
            n_cycles = panel[
                (panel["program"] == panel_prog)
                & (panel["domain"] == domain)
            ]["cycle"].nunique()
            print(
                f"\n=== {eval_prog}/{domain}: {n_countries} countries, "
                f"{n_cycles} cycles ===",
                flush=True,
            )
            rows = run_loco_program(
                panel, panel_prog, eval_prog, domain, score_col, model
            )
            print(f"  → {len(rows)} predictions", flush=True)
            all_preds.extend(rows)

    if not all_preds:
        raise RuntimeError("No predictions produced across any program/domain")

    pred_df  = pd.DataFrame(all_preds)
    fold_met = build_fold_metrics(pred_df)
    eq_fold  = build_equal_fold_summary(fold_met)

    OUT_DIR.mkdir(parents=True)
    pred_df.to_csv(OUT_DIR / "predictions.csv",         index=False, float_format="%.4f")
    fold_met.to_csv(OUT_DIR / "fold_metrics.csv",       index=False, float_format="%.4f")
    eq_fold.to_csv(OUT_DIR / "equal_fold_summary.csv",  index=False, float_format="%.4f")

    cache_ref = (
        Path.home()
        / ".cache/huggingface/hub/models--google--timesfm-2.5-200m-transformers/refs/main"
    )
    manifest = {
        "created_utc":    datetime.now(timezone.utc).isoformat(),
        "python":         platform.python_version(),
        "torch":          torch.__version__,
        "transformers":   importlib.metadata.version("transformers"),
        "device":         DEVICE,
        "model_id":       MODEL_ID,
        "cached_model_revision": (
            cache_ref.read_text().strip() if cache_ref.exists() else None
        ),
        "unified_panel":       str(UNIFIED.relative_to(ROOT)),
        "unified_panel_sha256": unified_hash,
        "programs_evaluated": {
            panel_prog: eval_prog
            for panel_prog, (eval_prog, _) in PROGRAM_CONFIGS.items()
        },
        "pisa_note": (
            "PISA sonuçları bu scriptten gelmez; "
            "run_pisa_timesfm25_ils_adapter.py → "
            "outputs/pisa_timesfm25_ils_adapter_v1/equal_fold_summary.csv"
        ),
        "evaluation_design": {
            "type":           "LOCO (Leave-One-Cycle-Out)",
            "context":        "Aynı program/alan/ülkenin önceki döngüleri",
            "min_context":    MIN_CONTEXT,
            "forecast_horizon": 1,
            "models":         ["TimesFM2_5_zero_shot"],
        },
        "outputs": [
            "predictions.csv",
            "fold_metrics.csv",
            "equal_fold_summary.csv",
        ],
    }
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== Equal-fold MAE per program/domain ===", flush=True)
    pivot = eq_fold.pivot_table(
        index=["program", "domain"],
        columns="model",
        values="equal_fold_MAE",
    )
    print(pivot.to_string(), flush=True)
    print(f"\nSaved to: {OUT_DIR.relative_to(ROOT)}/", flush=True)


if __name__ == "__main__":
    main()
