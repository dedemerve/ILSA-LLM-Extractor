#!/usr/bin/env python3
"""Frozen-configuration, outcome-exposed PISA 2025 TimesFM comparison.

The historical adaptation hyperparameters and epoch-selection rule are reused
unchanged. The PISA 2025 outcomes are used only to score forecasts in this
script; they are excluded from every ILSA training window. Since project users
have previously seen 2025 results, this is retrospective rather than blind.
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
from timesfm3 import ModelConfig, TimesFM3Evaluator
from transformers import TimesFm2_5ModelForPrediction

from run_pisa_timesfm25_ils_adapter import (
    BATCH_SIZE,
    DEVICE,
    DOMAINS,
    FORECAST_CONTEXT_LEN,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MAX_EPOCHS,
    MIN_CONTEXT,
    MODEL_ID as MODEL_25_ID,
    PATIENCE,
    SEED,
    add_lora,
    collect_series,
    forecasts,
    load_base_model,
    make_windows,
    select_epoch,
    train_epochs,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PISA_PANEL = ROOT / "outputs/stage5/pisa_forecasting_panel_v2.csv"
PISA_MANIFEST = ROOT / "outputs/stage5/pisa_forecasting_panel_v2_manifest.json"
UNIFIED_PANEL = ROOT / "outputs/stage5/unified_panel.csv"
TABULAR_DIR = ROOT / "outputs/pisa_2025_retrospective_tabular_v1"
OUT_DIR = ROOT / "outputs/pisa_2025_retrospective_timesfm_v1"
TARGET_CYCLE = 2025
MODEL_3_ID = "google/timesfm-3.0-pytorch"


def metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, model_name), group in frame.groupby(["domain", "model"], sort=True):
        error = group["prediction"].to_numpy(dtype=float) - group["true_score"].to_numpy(dtype=float)
        rows.append(
            {
                "target_cycle": TARGET_CYCLE,
                "domain": domain,
                "model": model_name,
                "n": len(group),
                "MAE": float(np.abs(error).mean()),
                "RMSE": float(np.sqrt(np.square(error).mean())),
                "mean_signed_error": float(error.mean()),
                "n_overpredicted": int((error > 0).sum()),
                "n_underpredicted": int((error < 0).sum()),
                "n_tied": int((error == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT_DIR}")
    for path in (PISA_PANEL, PISA_MANIFEST, UNIFIED_PANEL, TABULAR_DIR / "tabular_predictions.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    panel_hash = sha256(PISA_PANEL)
    panel_manifest = json.loads(PISA_MANIFEST.read_text(encoding="utf-8"))
    if panel_manifest["output_sha256"] != panel_hash:
        raise ValueError("PISA panel hash does not match its build manifest")

    pisa = pd.read_csv(PISA_PANEL, dtype={"country_iso3": str})
    unified = pd.read_csv(UNIFIED_PANEL, dtype={"country_iso3": str})
    series = collect_series(pisa, unified)
    tabular = pd.read_csv(TABULAR_DIR / "tabular_predictions.csv")
    countries_by_domain = {
        domain: set(tabular.loc[tabular["domain"].eq(domain), "country_iso3"])
        for domain in DOMAINS
    }
    if any(len(values) != 66 for values in countries_by_domain.values()):
        raise ValueError(f"Expected 66 eligible PISA 2025 countries per domain: {countries_by_domain}")
    common = set.intersection(*(countries_by_domain[domain] for domain in DOMAINS))
    if len(common) != 66:
        raise ValueError(f"Expected a 66-country common three-domain comparison; got {len(common)}")

    print("Selecting historical inner-validation epoch for the 2025 retrospective fold", flush=True)
    best_epoch, validation_record = select_epoch(series, TARGET_CYCLE, SEED + TARGET_CYCLE)
    training = make_windows(series, TARGET_CYCLE)
    if not training or any(window["target_cycle"] >= TARGET_CYCLE for window in training):
        raise ValueError("Invalid or empty pre-2025 ILSA adaptation window set")
    print(f"Refit on {len(training)} pre-2025 ILSA windows for {best_epoch} epochs", flush=True)
    adapted = add_lora(load_base_model(), SEED + TARGET_CYCLE)
    refit_losses = train_epochs(adapted, training, best_epoch, SEED + TARGET_CYCLE)
    zero_shot25 = load_base_model().eval()

    OUT_DIR.mkdir(parents=True)
    adapter_path = OUT_DIR / "adapter_target_2025"
    adapter_path.mkdir()
    adapted.save_pretrained(adapter_path)

    evaluator3 = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=MODEL_3_ID,
            per_core_batch_size=16,
            device="cpu",
            local_files_only=True,
        )
    )
    records = []
    for domain in DOMAINS:
        score_col = f"PISA_{domain}"
        subset = pisa.loc[
            pisa["domain"].eq(domain) & pisa[score_col].notna(),
            ["country_iso3", "cycle", score_col],
        ]
        contexts_25: list[np.ndarray] = []
        test_records: list[dict] = []
        for country in sorted(common):
            cdata = subset.loc[subset["country_iso3"].eq(country)].sort_values("cycle")
            context = cdata.loc[cdata["cycle"] < TARGET_CYCLE]
            target = cdata.loc[cdata["cycle"].eq(TARGET_CYCLE), score_col]
            if len(context) < 3 or len(target) != 1:
                raise ValueError(f"Invalid 2025 evaluation row: {country}/{domain}")
            record = {
                "target_cycle": TARGET_CYCLE,
                "domain": domain,
                "country_iso3": country,
                "n_context": len(context),
                "last_cycle": int(context["cycle"].iloc[-1]),
                "context_cycles": ";".join(context["cycle"].astype(int).astype(str)),
                "true_score": float(target.iloc[0]),
                "context": context[score_col].astype(float).to_numpy(dtype=np.float32),
                "target": float(target.iloc[0]),
            }
            test_records.append(record)
            contexts_25.append(record["context"])

        predictions_25_zero = forecasts(zero_shot25, test_records)
        predictions_25_adapter = forecasts(adapted, test_records)
        predictions_3 = list(
            evaluator3.predict_batch(
                contexts_25,
                horizon=1,
                return_quantiles=False,
                use_symmetric_averaging=True,
            )
        )
        if len(predictions_3) != len(test_records):
            raise ValueError(f"TimesFM 3.0 produced an unexpected count for {domain}")

        for row, value25_zero, value25_adapter, output3 in zip(
            test_records, predictions_25_zero, predictions_25_adapter, predictions_3
        ):
            pred3 = np.asarray(output3.forecast, dtype=float).reshape(-1)
            if pred3.size != 1 or not np.isfinite(pred3[0]):
                raise ValueError(f"Invalid TimesFM 3.0 forecast for {row['country_iso3']}/{domain}")
            for model_name, value in (
                ("TimesFM2_5_zero_shot", float(value25_zero)),
                ("TimesFM2_5_ILSA_LoRA", float(value25_adapter)),
                ("TimesFM3_zero_shot", float(pred3[0])),
            ):
                error = value - row["true_score"]
                records.append(
                    {
                        **{key: val for key, val in row.items() if key not in {"context", "target"}},
                        "model": model_name,
                        "prediction": value,
                        "signed_error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                    }
                )

    timesfm_predictions = pd.DataFrame(records)
    all_predictions = pd.concat([tabular, timesfm_predictions], ignore_index=True, sort=False)
    if all_predictions.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any():
        raise ValueError("Duplicate 2025 model prediction keys")
    if len(all_predictions) != 66 * 3 * 8:
        raise ValueError(f"Unexpected 2025 comparison row count: {len(all_predictions)}")
    metrics = metric_rows(all_predictions)
    if len(metrics) != 24 or metrics.n.ne(66).any() or not metrics.MAE.notna().all():
        raise ValueError("Incomplete 2025 retrospective metrics")

    timesfm_predictions.to_csv(OUT_DIR / "timesfm_predictions.csv", index=False)
    all_predictions.to_csv(OUT_DIR / "all_models_predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    cache3 = Path.home() / ".cache/huggingface/hub/models--google--timesfm-3.0-pytorch/refs/main"
    cache25 = Path.home() / ".cache/huggingface/hub/models--google--timesfm-2.5-200m-transformers/refs/main"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "timesfm_distribution": importlib.metadata.version("timesfm"),
        "device": "cpu",
        "target_cycle": TARGET_CYCLE,
        "role": "outcome-exposed retrospective evaluation; not a blind test",
        "exposure_disclosure": "PISA 2025 country-level outcomes and prior model results were available in this project before this run. No 2025 metrics were used to select training configuration in this run.",
        "common_country_count": len(common),
        "common_countries": sorted(common),
        "domains": list(DOMAINS),
        "minimum_pisa_context": 3,
        "pisa_panel": str(PISA_PANEL.relative_to(ROOT)),
        "pisa_panel_sha256": panel_hash,
        "unified_ilsa_panel": str(UNIFIED_PANEL.relative_to(ROOT)),
        "unified_ilsa_panel_sha256": sha256(UNIFIED_PANEL),
        "tabular_predictions": str((TABULAR_DIR / "tabular_predictions.csv").relative_to(ROOT)),
        "tabular_predictions_sha256": sha256(TABULAR_DIR / "tabular_predictions.csv"),
        "timesfm25_checkpoint": MODEL_25_ID,
        "timesfm25_revision": cache25.read_text().strip() if cache25.exists() else None,
        "timesfm3_checkpoint": MODEL_3_ID,
        "timesfm3_revision": cache3.read_text().strip() if cache3.exists() else None,
        "adaptation": {
            "validation": validation_record,
            "refit_windows": len(training),
            "selected_epochs": best_epoch,
            "normalized_train_loss_by_epoch": refit_losses,
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": ["q_proj", "v_proj"],
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "forecast_context_len": FORECAST_CONTEXT_LEN,
        },
        "pisa_2025_outcome_training_rule": "All ILSA training-pair target years are strictly < 2025. PISA 2025 values are used only as evaluation targets and are absent from contexts, training, and model selection.",
        "models": sorted(all_predictions.model.unique().tolist()),
        "outputs": ["timesfm_predictions.csv", "all_models_predictions.csv", "metrics.csv", "adapter_target_2025/"],
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(metrics.pivot(index="domain", columns="model", values="MAE").round(3).to_string())
    print(f"Predictions: {len(all_predictions)}; output: {OUT_DIR}")


if __name__ == "__main__":
    main()
