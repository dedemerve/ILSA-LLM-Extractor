#!/usr/bin/env python3
"""
ILSA Domain Fine-tuning v2 — Colab/GPU Runner
Protokol: outputs/finetuning/EVALUATION_PROTOCOL_v2.md

Bu script GPU ortamında (Google Colab A100/T4) çalıştırılmalıdır.
CPU'da çalışmaz.

Kullanım:
  1. Bu dosyayı Colab'a yükle
  2. !git clone <repo> && cd ILSA_LLMs
  3. !pip install -r requirements-finetuning-v2.txt
  4. !python3 scripts/finetuning/colab_lora_v2_training.py --arm qwen3 --lr 2e-4

Adımlar:
  Step 1: Preflight kontrolleri (build_sft_v2.py)
  Step 2: Qwen3-8B LoRA eğitimi (train_lora_v2.py)
  Step 3: Checkpoint seçimi (select_checkpoint.py)
  Step 4: Validation değerlendirmesi (extract_v2.py + eval_extraction.py)
  Step 5: Downstream (downstream_eval.py --split val)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n{'='*60}")
    print("CMD:", " ".join(cmd))
    print("="*60)
    result = subprocess.run(cmd, check=check, cwd=str(ROOT))
    return result


def check_gpu() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            sys.exit("HATA: GPU bulunamadı. Bu script sadece CUDA GPU'da çalışır.")
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 14:
            print("UYARI: Qwen3-8B QLoRA için en az 14 GB VRAM önerilir.")
    except ImportError:
        sys.exit("HATA: torch yüklü değil. pip install -r requirements-finetuning-v2.txt")


def step1_preflight(arm: str) -> None:
    """SFT veri setini oluştur ve preflight kontrolleri çalıştır."""
    print("\n[Step 1] SFT veri seti oluşturuluyor...")
    sft_train = ROOT / "outputs/finetuning/v2/data/sft_train.jsonl"
    if not sft_train.exists():
        run([sys.executable, "scripts/finetuning/build_sft_v2.py"])
    else:
        print(f"  SFT train verisi mevcut: {sft_train}")

    # Preflight kontrolleri
    model_id = "Qwen/Qwen3-8B" if arm == "qwen3" else "mistralai/Mistral-7B-Instruct-v0.3"
    preflight_out = ROOT / f"outputs/finetuning/v2/audit/preflight_v2_train_{arm}.json"
    if not preflight_out.exists():
        run([
            sys.executable, "scripts/finetuning/preflight_checks.py",
            "--model", model_id,
            "--split", "train",
            "--output", str(preflight_out),
        ])
    else:
        print(f"  Preflight sonucu mevcut: {preflight_out}")


def step2_train(arm: str, lr: float) -> Path:
    """LoRA v2 eğitimini başlat."""
    print(f"\n[Step 2] LoRA v2 eğitimi başlatılıyor (arm={arm}, lr={lr})...")
    run_dir = ROOT / f"outputs/finetuning/v2/runs/{arm}_r16_lr{int(lr*1e5)}"
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"  Run dizini mevcut: {run_dir}. Devam etmek için --force kullanın.")
        return run_dir

    model_id = "Qwen/Qwen3-8B" if arm == "qwen3" else "mistralai/Mistral-7B-Instruct-v0.3"
    run([
        sys.executable, "scripts/finetuning/train_lora_v2.py",
        "--model", model_id,
        "--output-dir", str(run_dir),
        "--lr", str(lr),
        "--epochs", "3",
        "--r", "16",
        "--lora-alpha", "32",
        "--dropout", "0.05",
        "--max-length", "8192",
    ])
    return run_dir


def step3_select_checkpoint(run_dir: Path, arm: str) -> Path:
    """Val strict F1'e göre en iyi checkpoint'i seç."""
    print(f"\n[Step 3] Checkpoint seçiliyor...")
    best_dir = run_dir / "best_checkpoint"
    if best_dir.exists():
        print(f"  Best checkpoint mevcut: {best_dir}")
        return best_dir

    # Her epoch'un adapterini val üzerinde değerlendir
    checkpoints = sorted(run_dir.glob("checkpoint-*"))
    if not checkpoints:
        sys.exit(f"HATA: {run_dir} içinde checkpoint bulunamadı.")

    print(f"  {len(checkpoints)} checkpoint bulundu.")
    model_id = "Qwen/Qwen3-8B" if arm == "qwen3" else "mistralai/Mistral-7B-Instruct-v0.3"

    best_f1 = -1.0
    best_ckpt = None
    for ckpt in checkpoints:
        arm_label = f"{arm}_lora_{ckpt.name}"
        pred_out = ROOT / f"outputs/finetuning/v2/predictions/{arm_label}_val.json"
        run([
            sys.executable, "scripts/finetuning/extract_v2.py",
            "--mode", "chunked",
            "--backend", "vllm",
            "--model", model_id,
            "--adapter", str(ckpt),
            "--split", "val",
            "--arm", arm_label,
        ])
        # F1'i oku
        eval_out = ROOT / f"outputs/finetuning/v2/eval/{arm_label}_val.json"
        run([
            sys.executable, "scripts/finetuning/eval_extraction.py",
            "--predictions", str(pred_out),
            "--output", str(eval_out),
        ])
        import json
        with open(eval_out) as f:
            d = json.load(f)
        arm_key = list(d.keys())[0]
        f1 = d[arm_key]["summary"]["strict"]["f1"]
        print(f"  {ckpt.name}: strict F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_ckpt = ckpt

    print(f"\n  En iyi checkpoint: {best_ckpt.name}, F1 = {best_f1:.4f}")
    best_ckpt.rename(best_dir)
    return best_dir


def step4_validate(best_ckpt: Path, arm: str) -> None:
    """Seçilen checkpoint'i validation seti üzerinde değerlendir."""
    print(f"\n[Step 4] Final validation değerlendirmesi...")
    arm_label = f"{arm}_lora_selected"
    pred_out = ROOT / f"outputs/finetuning/v2/predictions/{arm_label}_val.json"
    eval_out = ROOT / f"outputs/finetuning/v2/eval/{arm_label}_val.json"

    if eval_out.exists():
        print(f"  Eval mevcut: {eval_out}")
        return

    model_id = "Qwen/Qwen3-8B" if arm == "qwen3" else "mistralai/Mistral-7B-Instruct-v0.3"
    run([
        sys.executable, "scripts/finetuning/extract_v2.py",
        "--mode", "chunked",
        "--backend", "vllm",
        "--model", model_id,
        "--adapter", str(best_ckpt),
        "--split", "val",
        "--arm", arm_label,
    ])
    run([
        sys.executable, "scripts/finetuning/eval_extraction.py",
        "--predictions", str(pred_out),
        "--output", str(eval_out),
    ])


def step5_downstream(arm: str) -> None:
    """Downstream evaluation: W_j delta ve ΔMAE."""
    print(f"\n[Step 5] Downstream evaluation...")
    arm_label = f"{arm}_lora_selected"
    pred_file = ROOT / f"outputs/finetuning/v2/predictions/{arm_label}_val.json"
    out_file = ROOT / f"outputs/finetuning/v2/downstream/{arm_label}_val.json"

    if out_file.exists():
        print(f"  Downstream mevcut: {out_file}")
        return

    run([
        sys.executable, "scripts/finetuning/downstream_eval.py",
        "--predictions", str(pred_file),
        "--split", "val",
        "--mode", "augment",
        "--output", str(out_file),
    ])


def print_summary(arm: str) -> None:
    """Sonuçları konsola yazdır."""
    import json
    arm_label = f"{arm}_lora_selected"
    eval_file = ROOT / f"outputs/finetuning/v2/eval/{arm_label}_val.json"
    downstream_file = ROOT / f"outputs/finetuning/v2/downstream/{arm_label}_val.json"

    print("\n" + "="*60)
    print("ÖZET")
    print("="*60)

    if eval_file.exists():
        with open(eval_file) as f:
            d = json.load(f)
        key = list(d.keys())[0]
        s = d[key]["summary"]
        print(f"\nExtraction ({arm_label}):")
        print(f"  Strict pair F1:  {s['strict']['f1']:.4f}")
        print(f"  Lenient pair F1: {s['lenient']['f1']:.4f}")
        print(f"  Macro strict F1: {s['macro_strict_f1']:.4f}")
        print(f"  N pred: {s['n_pred']}, N gold: {s['n_gold']}")

    if downstream_file.exists():
        with open(downstream_file) as f:
            d = json.load(f)
        mae = d.get("mae", {})
        dw = d.get("delta_w", {})
        print(f"\nDownstream:")
        print(f"  Oracle MAE:       {mae.get('oracle', '—')}")
        print(f"  {arm_label} MAE: {mae.get(arm_label, '—')}")
        print(f"  Stage5 frozen MAE:{mae.get('stage5_frozen', '—')}")
        arm_dw = dw.get(arm_label, {})
        print(f"  ΔW L1:            {arm_dw.get('l1_to_oracle', '—')}")
        print(f"  ΔW Spearman ρ:    {arm_dw.get('spearman_to_oracle', '—')}")

    # Oracle ile karşılaştır
    oracle_eval = ROOT / "outputs/finetuning/v2/eval/oracle_ceiling_val.json"
    if oracle_eval.exists():
        with open(oracle_eval) as f:
            d = json.load(f)
        oracle_f1 = d["oracle"]["summary"]["strict"]["f1"]
        print(f"\nOracle ceiling (val): strict F1 = {oracle_f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ILSA LoRA v2 Training Pipeline")
    parser.add_argument("--arm", choices=["qwen3", "mistral"], default="qwen3",
                        help="Hangi model kullanılacak")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (1e-4 veya 2e-4)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Eğitimi atla (mevcut checkpoint'lerden devam)")
    parser.add_argument("--force", action="store_true",
                        help="Mevcut run dizinini sil ve baştan başla")
    args = parser.parse_args()

    print("ILSA LoRA v2 Training Pipeline")
    print(f"Arm: {args.arm}, LR: {args.lr}")

    check_gpu()
    step1_preflight(args.arm)

    if not args.skip_train:
        run_dir = step2_train(args.arm, args.lr)
    else:
        arm_key = f"{args.arm}_r16_lr{int(args.lr*1e5)}"
        run_dir = ROOT / f"outputs/finetuning/v2/runs/{arm_key}"
        if not run_dir.exists():
            sys.exit(f"HATA: --skip-train belirtildi ama run dizini yok: {run_dir}")

    best_ckpt = step3_select_checkpoint(run_dir, args.arm)
    step4_validate(best_ckpt, args.arm)
    step5_downstream(args.arm)
    print_summary(args.arm)


if __name__ == "__main__":
    main()
