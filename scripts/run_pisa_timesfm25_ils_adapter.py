#!/usr/bin/env python3
"""Temporal validation of TimesFM 2.5 zero-shot and ILSA LoRA adaptation.

Outer targets are historical PISA cycles 2012, 2015, 2018, and 2022. Training
windows pool prior PISA, TIMSS, and PIRLS country-level score series; PISA 2025
is never a training target or model-selection signal. Each outer fold selects
an epoch count on the latest earlier ILSA target cycle, then refits an adapter
on all eligible pre-outer-fold windows for that fixed number of epochs.
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
from peft import LoraConfig, get_peft_model
from transformers import TimesFm2_5ModelForPrediction


ROOT = Path(__file__).resolve().parents[1]
PISA_PANEL = ROOT / "outputs/stage5/pisa_forecasting_panel_v2.csv"
PISA_MANIFEST = ROOT / "outputs/stage5/pisa_forecasting_panel_v2_manifest.json"
UNIFIED_PANEL = ROOT / "outputs/stage5/unified_panel.csv"
FOLDS_PATH = ROOT / "analysis/03_data_feasibility/audit/pisa_historical_folds_v1.json"
TABULAR_PREDICTIONS = ROOT / "outputs/pisa_tabular_benchmark_v1/historical_predictions.csv"
TIMESFM3_PREDICTIONS = ROOT / "outputs/pisa_timesfm_historical_zeroshot_v2/timesfm_predictions.csv"
OUT_DIR = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1"
MODEL_ID = "google/timesfm-2.5-200m-transformers"
DOMAINS = ("mathematics", "reading", "science")
MIN_CONTEXT = 2
FORECAST_CONTEXT_LEN = 32
BATCH_SIZE = 8
MAX_EPOCHS = 4
PATIENCE = 2
LEARNING_RATE = 5e-5
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
SEED = 20261015
DEVICE = "cpu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_series(pisa_panel: pd.DataFrame, unified: pd.DataFrame) -> list[dict]:
    """Return independent program/domain/grade/country score sequences."""
    specifications = [
        (pisa_panel, "PISA", "mathematics", "PISA_mathematics"),
        (pisa_panel, "PISA", "reading", "PISA_reading"),
        (pisa_panel, "PISA", "science", "PISA_science"),
        (unified, "TIMSS", "mathematics", "TIMSS_mathematics"),
        (unified, "TIMSS", "science", "TIMSS_science"),
        (unified, "TIMSS_G4", "mathematics", "TIMSS_G4_mathematics"),
        (unified, "TIMSS_G4", "science", "TIMSS_G4_science"),
        (unified, "PIRLS", "reading", "PIRLS_reading"),
    ]
    result: list[dict] = []
    for frame, program, domain, score_col in specifications:
        # unified_panel repeats joined score columns on rows from other
        # assessments; restrict to the source assessment/domain before grouping.
        source_rows = frame["program"].eq(program) & frame["domain"].eq(domain)
        subset = frame.loc[
            source_rows & frame[score_col].notna(), ["country_iso3", "cycle", score_col]
        ].copy()
        subset["cycle"] = pd.to_numeric(subset["cycle"], errors="raise").astype(int)
        if subset.duplicated(["country_iso3", "cycle"]).any():
            duplicate = subset.loc[
                subset.duplicated(["country_iso3", "cycle"], keep=False),
                ["country_iso3", "cycle"],
            ].head(5)
            raise ValueError(f"Duplicate source series keys in {score_col}:\n{duplicate}")
        for country, group in subset.groupby("country_iso3", sort=True):
            group = group.sort_values("cycle")
            years = group["cycle"].astype(int).tolist()
            values = group[score_col].astype(float).tolist()
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite score in {program}/{domain}/{country}")
            if len(values) >= MIN_CONTEXT + 1:
                result.append(
                    {
                        "series_id": f"{program}|{domain}|{country}",
                        "program": program,
                        "domain": domain,
                        "country_iso3": str(country),
                        "years": years,
                        "values": values,
                    }
                )
    return result


def make_windows(series: list[dict], cutoff: int) -> list[dict]:
    """Create one expanding-context training pair per eligible past target."""
    windows: list[dict] = []
    for item in series:
        retained = [(y, v) for y, v in zip(item["years"], item["values"]) if y < cutoff]
        years = [entry[0] for entry in retained]
        values = [entry[1] for entry in retained]
        for index in range(MIN_CONTEXT, len(years)):
            windows.append(
                {
                    "series_id": item["series_id"],
                    "target_cycle": int(years[index]),
                    "context": np.asarray(values[:index], dtype=np.float32),
                    "target": float(values[index]),
                }
            )
    return windows


def validation_cycle(series: list[dict], outer_target: int) -> int:
    outer_windows = make_windows(series, outer_target)
    windows_by_cycle: dict[int, int] = {}
    for example in outer_windows:
        windows_by_cycle[example["target_cycle"]] = windows_by_cycle.get(example["target_cycle"], 0) + 1
    candidates = sorted(
        windows_by_cycle,
        reverse=True,
    )
    for candidate in candidates:
        if windows_by_cycle[candidate]:
            return candidate
    raise ValueError(f"No temporally valid validation cycle before {outer_target}")


def load_base_model() -> TimesFm2_5ModelForPrediction:
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    return model.to(DEVICE)


def add_lora(model: TimesFm2_5ModelForPrediction, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    adapted = get_peft_model(model, config)
    adapted.print_trainable_parameters()
    return adapted


def batches(examples: list[dict], batch_size: int, shuffle_seed: int | None = None):
    indexes = np.arange(len(examples))
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(indexes)
    for start in range(0, len(indexes), batch_size):
        yield [examples[index] for index in indexes[start : start + batch_size]]


def tensor_batch(batch: list[dict]):
    past = [torch.as_tensor(row["context"], dtype=torch.float32, device=DEVICE) for row in batch]
    future = torch.as_tensor(
        [[row["target"]] for row in batch], dtype=torch.float32, device=DEVICE
    )
    return past, future


@torch.no_grad()
def evaluate_mae(model, examples: list[dict]) -> float:
    model.eval()
    absolute_errors: list[float] = []
    for batch in batches(examples, BATCH_SIZE):
        past, future = tensor_batch(batch)
        output = model(
            past_values=past,
            future_values=future,
            forecast_context_len=FORECAST_CONTEXT_LEN,
            truncate_negative=False,
        )
        forecasts = output.mean_predictions[:, 0].detach().cpu().numpy()
        targets = future[:, 0].detach().cpu().numpy()
        absolute_errors.extend(np.abs(forecasts - targets).tolist())
    if not absolute_errors or not np.isfinite(absolute_errors).all():
        raise ValueError("Validation produced no finite forecasts")
    return float(np.mean(absolute_errors))


def train_epochs(model, examples: list[dict], epochs: int, seed: int) -> list[float]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("LoRA adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for batch in batches(examples, BATCH_SIZE, seed + epoch):
            past, future = tensor_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                past_values=past,
                future_values=future,
                forecast_context_len=FORECAST_CONTEXT_LEN,
                truncate_negative=False,
            )
            if output.loss is None or not torch.isfinite(output.loss):
                raise ValueError("Fine-tuning returned a missing or non-finite loss")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(output.loss.detach().cpu()))
        if not losses:
            raise ValueError("No optimization batches were produced")
        history.append(float(np.mean(losses)))
        print(f"    epoch {epoch + 1}/{epochs}: normalized_loss={history[-1]:.6f}", flush=True)
    return history


def select_epoch(series: list[dict], outer_target: int, seed: int) -> tuple[int, dict]:
    val_cycle = validation_cycle(series, outer_target)
    training = [
        example for example in make_windows(series, val_cycle)
        if example["target_cycle"] < val_cycle
    ]
    validation = [
        example for example in make_windows(series, outer_target)
        if example["target_cycle"] == val_cycle
    ]
    if not training or not validation:
        raise ValueError(
            f"Insufficient nested temporal data for {outer_target}: "
            f"train={len(training)}, validation={len(validation)} at {val_cycle}"
        )
    print(
        f"  Outer {outer_target}: inner validation cycle {val_cycle}; "
        f"train windows={len(training)}, validation windows={len(validation)}",
        flush=True,
    )
    model = add_lora(load_base_model(), seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    epoch_log = []
    for epoch in range(MAX_EPOCHS):
        losses: list[float] = []
        model.train()
        for batch in batches(training, BATCH_SIZE, seed + epoch):
            past, future = tensor_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                past_values=past,
                future_values=future,
                forecast_context_len=FORECAST_CONTEXT_LEN,
                truncate_negative=False,
            )
            if output.loss is None or not torch.isfinite(output.loss):
                raise ValueError("Fine-tuning returned a missing or non-finite loss")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(output.loss.detach().cpu()))
        if not losses:
            raise ValueError("No inner-training batches were produced")
        val_mae = evaluate_mae(model, validation)
        epoch_log.append(
            {"epoch": epoch + 1, "train_normalized_loss": float(np.mean(losses)), "validation_MAE": val_mae}
        )
        print(
            f"    selection epoch {epoch + 1}/{MAX_EPOCHS}: "
            f"loss={np.mean(losses):.6f}, validation_MAE={val_mae:.4f}",
            flush=True,
        )
        if val_mae < best_mae - 1e-8:
            best_mae = val_mae
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    del model, optimizer
    if best_epoch < 1:
        raise ValueError(f"No best epoch selected for outer target {outer_target}")
    return best_epoch, {
        "validation_cycle": val_cycle,
        "training_windows_for_selection": len(training),
        "validation_windows": len(validation),
        "best_validation_MAE": best_mae,
        "selected_epoch_count": best_epoch,
        "epoch_log": epoch_log,
    }


@torch.no_grad()
def forecasts(model, examples: list[dict]) -> np.ndarray:
    model.eval()
    result: list[float] = []
    for batch in batches(examples, BATCH_SIZE):
        past, _ = tensor_batch(batch)
        output = model(
            past_values=past,
            forecast_context_len=FORECAST_CONTEXT_LEN,
            truncate_negative=False,
        )
        result.extend(output.mean_predictions[:, 0].detach().cpu().numpy().astype(float).tolist())
    values = np.asarray(result, dtype=float)
    if len(values) != len(examples) or not np.isfinite(values).all():
        raise ValueError("Forecast length mismatch or non-finite forecast")
    return values


def score_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target_cycle, domain, model_name), group in predictions.groupby(
        ["target_cycle", "domain", "model"], sort=True
    ):
        error = group["prediction"].to_numpy(float) - group["true_score"].to_numpy(float)
        rows.append(
            {
                "target_cycle": int(target_cycle),
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
    for path in (PISA_PANEL, PISA_MANIFEST, UNIFIED_PANEL, FOLDS_PATH, TABULAR_PREDICTIONS, TIMESFM3_PREDICTIONS):
        if not path.is_file():
            raise FileNotFoundError(path)

    pisa_hash = sha256(PISA_PANEL)
    panel_manifest = json.loads(PISA_MANIFEST.read_text(encoding="utf-8"))
    if panel_manifest["output_sha256"] != pisa_hash:
        raise ValueError("Canonical PISA panel hash does not match its manifest")
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    if folds["panel_sha256"] != pisa_hash:
        raise ValueError("Frozen folds refer to a different canonical PISA panel")
    if folds["pisa_2025"]["excluded_from_historical_fold_manifest"] is not True:
        raise ValueError("PISA 2025 must remain excluded from historical development")

    pisa = pd.read_csv(PISA_PANEL, dtype={"country_iso3": str})
    unified = pd.read_csv(UNIFIED_PANEL, dtype={"country_iso3": str})
    series = collect_series(pisa, unified)
    if not series:
        raise ValueError("No eligible ILSA country-level series found")
    print(f"Eligible ILSA series: {len(series)}", flush=True)

    OUT_DIR.mkdir(parents=True)
    (OUT_DIR / "adapters").mkdir()
    all_predictions: list[dict] = []
    fold_runs: list[dict] = []
    tabular = pd.read_csv(TABULAR_PREDICTIONS)
    timesfm3 = pd.read_csv(TIMESFM3_PREDICTIONS)

    for fold in folds["folds"]:
        target_cycle = int(fold["target_cycle"])
        countries = sorted(fold["common_three_domain_countries"])
        print(f"\n=== Outer PISA target {target_cycle} ===", flush=True)
        chosen_epochs, validation_record = select_epoch(series, target_cycle, SEED + target_cycle)
        full_training = make_windows(series, target_cycle)
        if not full_training:
            raise ValueError(f"No adaptation windows before {target_cycle}")
        print(f"  Refit on {len(full_training)} ILSA windows for {chosen_epochs} epochs", flush=True)
        adapted = add_lora(load_base_model(), SEED + target_cycle)
        final_loss = train_epochs(adapted, full_training, chosen_epochs, SEED + target_cycle)
        adapter_dir = OUT_DIR / "adapters" / f"target_{target_cycle}"
        adapter_dir.mkdir()
        adapted.save_pretrained(adapter_dir)

        for domain in DOMAINS:
            score_col = f"PISA_{domain}"
            subset = pisa.loc[
                pisa["domain"].eq(domain) & pisa[score_col].notna(),
                ["country_iso3", "cycle", score_col],
            ]
            test_examples = []
            for country in countries:
                cdata = subset.loc[subset["country_iso3"].eq(country)].sort_values("cycle")
                context = cdata.loc[cdata["cycle"] < target_cycle]
                target = cdata.loc[cdata["cycle"].eq(target_cycle), score_col]
                if len(context) < 3 or len(target) != 1:
                    raise ValueError(
                        f"Frozen fold mismatch at {target_cycle}/{domain}/{country}: "
                        f"context={len(context)}, target={len(target)}"
                    )
                test_examples.append(
                    {
                        "target_cycle": target_cycle,
                        "domain": domain,
                        "country_iso3": country,
                        "n_context": len(context),
                        "last_cycle": int(context["cycle"].iloc[-1]),
                        "true_score": float(target.iloc[0]),
                        "context_cycles": ";".join(context["cycle"].astype(int).astype(str)),
                        "context": context[score_col].astype(float).to_numpy(dtype=np.float32),
                        "target": float(target.iloc[0]),
                    }
                )

            # Evaluate the same TimesFM 2.5 checkpoint with no trainable adapters.
            zero_shot_model = load_base_model().eval()
            zero_shot_values = forecasts(zero_shot_model, test_examples)
            del zero_shot_model
            adapted_values = forecasts(adapted, test_examples)
            for record, zero_shot, adapted_value in zip(test_examples, zero_shot_values, adapted_values):
                for model_name, value in (
                    ("TimesFM2_5_zero_shot", float(zero_shot)),
                    ("TimesFM2_5_ILSA_LoRA", float(adapted_value)),
                ):
                    error = value - record["true_score"]
                    all_predictions.append(
                        {
                            **{key: val for key, val in record.items() if key not in {"context", "target"}},
                            "model": model_name,
                            "prediction": value,
                            "signed_error": error,
                            "absolute_error": abs(error),
                            "squared_error": error**2,
                        }
                    )
        fold_runs.append(
            {
                "outer_target_cycle": target_cycle,
                **validation_record,
                "refit_training_windows": len(full_training),
                "refit_train_loss_by_epoch": final_loss,
                "adapter_path": str(adapter_dir.relative_to(ROOT)),
            }
        )
        del adapted

    learned = pd.DataFrame(all_predictions)
    if learned.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any():
        raise ValueError("Duplicate adapter or TimesFM 2.5 zero-shot forecast keys")
    main_predictions = pd.concat([tabular, timesfm3, learned], ignore_index=True, sort=False)
    metrics = score_frame(main_predictions)
    equal_fold = (
        metrics.groupby(["domain", "model"], as_index=False)
        .agg(
            equal_fold_MAE=("MAE", "mean"),
            equal_fold_RMSE=("RMSE", "mean"),
            equal_fold_mean_signed_error=("mean_signed_error", "mean"),
            folds=("target_cycle", "nunique"),
        )
    )
    learned.to_csv(OUT_DIR / "timesfm25_adaptation_predictions.csv", index=False)
    main_predictions.to_csv(OUT_DIR / "all_models_historical_predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "historical_metrics.csv", index=False)
    equal_fold.to_csv(OUT_DIR / "equal_fold_summary.csv", index=False)

    cache_ref = Path.home() / ".cache/huggingface/hub/models--google--timesfm-2.5-200m-transformers/refs/main"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "device": DEVICE,
        "model_id": MODEL_ID,
        "cached_model_revision": cache_ref.read_text().strip() if cache_ref.exists() else None,
        "model_use": "TimesFM 2.5 used for both zero-shot and LoRA-adapted arms; TimesFM 3.0 is reported as a separate family comparator",
        "pisa_panel": str(PISA_PANEL.relative_to(ROOT)),
        "pisa_panel_sha256": pisa_hash,
        "unified_ilsa_panel": str(UNIFIED_PANEL.relative_to(ROOT)),
        "unified_ilsa_panel_sha256": sha256(UNIFIED_PANEL),
        "fold_manifest": str(FOLDS_PATH.relative_to(ROOT)),
        "fold_manifest_sha256": sha256(FOLDS_PATH),
        "folds": fold_runs,
        "ilsa_training_sources": sorted({item["program"] for item in series}),
        "n_ilsa_country_domain_series": len(series),
        "window_rule": "All expanding-prefix pairs with at least two prior values; every target is strictly earlier than the applicable validation or outer-test origin.",
        "validation_rule": "For each outer fold, select epoch count using the latest earlier cycle with eligible ILSA target windows; train selection model only on target cycles strictly before that inner validation cycle; refit fresh adapter on all windows strictly before outer target using selected epoch count.",
        "training_config": {
            "minimum_context": MIN_CONTEXT,
            "forecast_context_len": FORECAST_CONTEXT_LEN,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.01,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_target_modules": ["q_proj", "v_proj"],
            "lora_dropout": LORA_DROPOUT,
            "seed": SEED,
        },
        "pisa_2025": "Not included as an outer test in this adaptation run and never used for model selection; any separate use remains outcome-exposed retrospective analysis.",
        "existing_baseline_prediction_sha256": sha256(TABULAR_PREDICTIONS),
        "existing_timesfm3_prediction_sha256": sha256(TIMESFM3_PREDICTIONS),
        "outputs": [
            "timesfm25_adaptation_predictions.csv",
            "all_models_historical_predictions.csv",
            "historical_metrics.csv",
            "equal_fold_summary.csv",
            "adapters/target_<cycle>/",
        ],
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nEqual-fold MAE across historical outer folds:", flush=True)
    print(equal_fold.pivot(index="domain", columns="model", values="equal_fold_MAE").to_string(), flush=True)


if __name__ == "__main__":
    main()
