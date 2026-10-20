"""
ADIM 3 — TimesFM 2.5 LoRA Fine-Tuning

KAVRAM:
  Fine-tuning = genel bir modeli kendi verilerimizle özelleştirmek.
  LoRA = modelin yalnızca küçük bir kısmını eğitmek (~%0.6 parametre).
  Bu sayede büyük bir modeli az veriyle, hızlıca adapte edebiliriz.

NE YAPIYORUZ:
  1. TimesFM 2.5 (200M parametre) yükle
  2. LoRA adapter ekle (sadece ~1.4M parametre eğitilecek)
  3. Eğitim verisi: PISA + TIMSS + PIRLS zaman serileri (2022 öncesi)
  4. 2022 test seti üzerinde değerlendir
  5. Persistence ve zero-shot ile karşılaştır

EĞİTİM MANTIĞI:
  Her adımda seriden rastgele bir (context, horizon) penceresi alınır.
  Örnek: PISA Türkiye serisinden [2003,2006,2009] → [2012] tahmini
  Model bu örneklerden "ILSA serilerinin nasıl davrandığını" öğrenir.

ÇALIŞTIR:
  python3 scripts/step3_finetune_timesfm.py

ÇIKTILAR:
  outputs/step3/ilsa-lora/          — fine-tuned model
  outputs/step3/finetuned_results.csv
  outputs/step3/comparison_table.csv — 3 modelin karşılaştırması
  outputs/step3/comparison_bar.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

ROOT    = Path(__file__).resolve().parents[1]
PANEL   = ROOT / "outputs" / "stage5" / "unified_panel.csv"
OUT_DIR = ROOT / "outputs" / "step3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Sabitler ---
TEST_CYCLE   = 2022
MIN_CONTEXT  = 3      # eğitim için minimum context uzunluğu
HORIZON      = 1      # bir adım ilerisi
CONTEXT_LEN  = 5      # fine-tuning pencere boyutu (sabit)
NUM_WINDOWS  = 150    # hızlı kanıt: 400 → 150
EPOCHS       = 3     # hızlı kanıt: 8 → 3
BATCH_SIZE   = 4
LR           = 1e-4
LORA_R       = 4      # LoRA rank: düşük = az parametre, hızlı
LORA_ALPHA   = 8
SEED         = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Cihaz: {device}\n")

# =============================================================================
# BÖLÜM 1: Eğitim verisi hazırla
# =============================================================================
print("Eğitim verisi hazırlanıyor...")
panel = pd.read_csv(PANEL)

# Eğitimde kullanacağımız program-domain-sütun üçlüleri
TRAIN_SOURCES = [
    ("PISA",  "mathematics", "PISA_mathematics"),
    ("PISA",  "reading",     "PISA_reading"),
    ("PISA",  "science",     "PISA_science"),
    ("TIMSS", "mathematics", "TIMSS_mathematics"),
    ("TIMSS", "science",     "TIMSS_science"),
    ("PIRLS", "reading",     "PIRLS_reading"),
]

# Her program-domain için ülke zaman serilerini topla
# Kural: TEST_CYCLE'dan önceki döngüler kullanılabilir
all_series = []
for prog, dom, col in TRAIN_SOURCES:
    sub = panel[(panel["program"] == prog) & (panel["domain"] == dom)].copy()
    sub = sub[sub["cycle"] < TEST_CYCLE].dropna(subset=[col])
    sub = sub.sort_values(["country_iso3", "cycle"])
    for country, grp in sub.groupby("country_iso3"):
        scores = grp[col].tolist()
        if len(scores) >= MIN_CONTEXT:
            all_series.append(scores)

print(f"  Toplam seri: {len(all_series)}")
print(f"  Ortalama uzunluk: {np.mean([len(s) for s in all_series]):.1f} döngü")

# Rastgele (context, hedef) pencereleri oluştur
# Her pencere: CONTEXT_LEN geçmiş değer → sonraki 1 değer
windows = []
for _ in range(NUM_WINDOWS):
    series = all_series[np.random.randint(len(all_series))]
    if len(series) < CONTEXT_LEN + HORIZON:
        # Kısa seriler için tüm seriyi kullan
        if len(series) < 2:
            continue
        cut = np.random.randint(1, len(series))
        ctx = series[:cut]
        tgt = series[cut:cut + HORIZON]
    else:
        start = np.random.randint(0, len(series) - CONTEXT_LEN - HORIZON + 1)
        ctx = series[start:start + CONTEXT_LEN]
        tgt = series[start + CONTEXT_LEN:start + CONTEXT_LEN + HORIZON]
    if len(tgt) == HORIZON:
        windows.append((ctx, tgt))

print(f"  Eğitim penceresi: {len(windows)}\n")


class ILSADataset(Dataset):
    """Her örnek: sabit uzunluğa pad edilmiş context + hedef değerler."""

    def __init__(self, windows, context_len):
        self.windows = windows
        self.context_len = context_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ctx, tgt = self.windows[idx]
        # Context'i context_len'e pad et (soldan sıfır)
        padded = np.zeros(self.context_len, dtype=np.float32)
        ctx_arr = np.array(ctx, dtype=np.float32)
        padded[-len(ctx_arr):] = ctx_arr
        return {
            "past_values":   torch.tensor(padded,           dtype=torch.float32),
            "future_values": torch.tensor(tgt,              dtype=torch.float32),
            "freq_token":    torch.tensor([0],              dtype=torch.long),    # 0 = yüksek frekanssız
        }


dataset    = ILSADataset(windows, CONTEXT_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================================================================
# BÖLÜM 2: Model ve LoRA
# =============================================================================
print("TimesFM 2.5 yükleniyor...")
MODEL_ID = "google/timesfm-2.5-200m-transformers"

try:
    from transformers import TimesFm2_5ModelForPrediction
    model = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
except Exception as e:
    print(f"  HATA: {e}")
    print("  Not: TimesFM 2.5 transformers entegrasyonu için transformers>=4.52 gerekebilir.")
    raise

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model yüklendi. Toplam parametre: {total_params:,}")

# LoRA konfigürasyonu
# target_modules="all-linear": tüm lineer katmanlara LoRA uygula
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  LoRA eklendi. Eğitilecek parametre: {trainable:,} "
      f"({100*trainable/total_params:.2f}%)\n")

model = model.to(device)

# =============================================================================
# BÖLÜM 3: Eğitim
# =============================================================================
print(f"Fine-tuning başlıyor ({EPOCHS} epoch, batch={BATCH_SIZE}, lr={LR})...")
print("Her epoch yaklaşık 10-30 saniye sürer (MPS).\n")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=LR
)

epoch_losses = []
# PEFT sarmalayıcısı attention_mask çakışmasına yol açıyor.
# Çözüm: PEFT'in base_model'ini doğrudan kullan — gradyanlar hâlâ LoRA
# ağırlıklarına akar, eğitim aynı şekilde çalışır.
base = model.base_model.model

model.train()
for epoch in range(EPOCHS):
    batch_losses = []
    for batch in dataloader:
        past   = batch["past_values"].to(device)
        future = batch["future_values"].to(device)
        freq   = batch["freq_token"].to(device)

        optimizer.zero_grad()
        out = base(
            past_values=past,
            future_values=future,
            freq_token=freq,
        )
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_losses.append(loss.item())

    epoch_loss = np.mean(batch_losses)
    epoch_losses.append(epoch_loss)
    print(f"  Epoch {epoch+1:2d}/{EPOCHS}  loss={epoch_loss:.4f}")

print("\nEğitim tamamlandı.")

# Adapter'ı kaydet
adapter_dir = OUT_DIR / "ilsa-lora"
model.save_pretrained(adapter_dir)
print(f"  → Adapter kaydedildi: {adapter_dir}\n")

# =============================================================================
# BÖLÜM 4: Değerlendirme — PISA Matematik 2022
# =============================================================================
print("Fine-tuned model değerlendiriliyor...")
print("Test: PISA matematik 2022 (eğitimde hiç görülmedi)\n")

# Test verisi
sub_pisa = panel[(panel["program"] == "PISA") & (panel["domain"] == "mathematics")].copy()
sub_pisa = sub_pisa.dropna(subset=["PISA_mathematics"]).sort_values(["country_iso3", "cycle"])

test_records = []
for country, grp in sub_pisa.groupby("country_iso3"):
    ctx_rows = grp[grp["cycle"] < TEST_CYCLE]
    tgt_rows = grp[grp["cycle"] == TEST_CYCLE]
    if len(ctx_rows) < MIN_CONTEXT and len(tgt_rows) == 0:
        continue
    if len(tgt_rows) == 0:
        continue
    test_records.append({
        "country":     country,
        "ctx_scores":  ctx_rows["PISA_mathematics"].tolist(),
        "true_score":  float(tgt_rows["PISA_mathematics"].iloc[0]),
        "persistence": float(ctx_rows["PISA_mathematics"].iloc[-1]) if len(ctx_rows) > 0 else np.nan,
    })

# Fine-tuned model ile tahmin
model.eval()
ft_preds = []
with torch.no_grad():
    for rec in test_records:
        ctx = np.array(rec["ctx_scores"], dtype=np.float32)
        padded = np.zeros(CONTEXT_LEN, dtype=np.float32)
        padded[-min(len(ctx), CONTEXT_LEN):] = ctx[-min(len(ctx), CONTEXT_LEN):]
        past   = torch.tensor(padded).unsqueeze(0).to(device)
        freq   = torch.tensor([[0]], dtype=torch.long).to(device)
        out    = base(past_values=past, freq_token=freq)
        # out.prediction_outputs: (batch, horizon, ...) — median forecast
        pred_val = float(out.prediction_outputs[0, 0].cpu())
        ft_preds.append(pred_val)

# Sıfır-shot için önceki adımdaki sonuçları yükle
zeroshot_csv = ROOT / "outputs" / "step2" / "zeroshot_results.csv"
zs_df = pd.read_csv(zeroshot_csv) if zeroshot_csv.exists() else None

# Sonuç tablosu
results = []
for rec, ft_pred in zip(test_records, ft_preds):
    row = {
        "country":    rec["country"],
        "true_score": rec["true_score"],
        "persistence": rec["persistence"],
        "timesfm_finetuned": ft_pred,
        "err_persistence": abs(rec["true_score"] - rec["persistence"]) if not np.isnan(rec["persistence"]) else np.nan,
        "err_finetuned":   abs(rec["true_score"] - ft_pred),
    }
    # zero-shot varsa ekle
    if zs_df is not None:
        zs_row = zs_df[zs_df["country"] == rec["country"]]
        if len(zs_row) > 0:
            row["timesfm_zeroshot"]  = float(zs_row["timesfm_zeroshot"].iloc[0])
            row["err_zeroshot"]      = float(zs_row["err_timesfm"].iloc[0])
    results.append(row)

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "finetuned_results.csv", index=False)

# =============================================================================
# BÖLÜM 5: Karşılaştırma tablosu
# =============================================================================
print("=" * 60)
print("KARŞILAŞTIRMA: PISA Matematik 2022")
print("=" * 60)

# Sadece her üç modelin de mevcut olduğu ülkeler
compare_cols = ["err_persistence", "err_finetuned"]
if "err_zeroshot" in df.columns:
    compare_cols.append("err_zeroshot")
compare = df.dropna(subset=compare_cols)

comparison_rows = [
    {"model": "Persistence (baseline)",  "MAE": compare["err_persistence"].mean(), "n": len(compare)},
    {"model": "TimesFM Fine-tuned (LoRA)", "MAE": compare["err_finetuned"].mean(),  "n": len(compare)},
]
if "err_zeroshot" in compare.columns:
    comparison_rows.insert(1, {
        "model": "TimesFM Zero-shot",
        "MAE":   compare["err_zeroshot"].mean(),
        "n":     len(compare),
    })

comp_df = pd.DataFrame(comparison_rows)
comp_df["MAE"] = comp_df["MAE"].round(2)
comp_df["vs_persistence"] = (comp_df["MAE"] - comp_df.loc[0, "MAE"]).round(2)
comp_df.to_csv(OUT_DIR / "comparison_table.csv", index=False)

print()
for _, row in comp_df.iterrows():
    delta = row["vs_persistence"]
    if delta == 0:
        tag = "(baseline)"
    elif delta < 0:
        tag = f"↓ {abs(delta):.2f} daha iyi"
    else:
        tag = f"↑ {delta:.2f} daha kötü"
    print(f"  {row['model']:35} MAE={row['MAE']:.2f}  {tag}")

# =============================================================================
# BÖLÜM 6: Görsel
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PISA Matematik 2022 — Fine-Tuning Sonuçları", fontsize=13)

# Sol: loss eğrisi
axes[0].plot(range(1, EPOCHS+1), epoch_losses, "o-", color="#2563eb", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss")
axes[0].set_title("Fine-Tuning Loss Eğrisi")
axes[0].grid(True, alpha=0.3)

# Sağ: MAE bar karşılaştırması
models  = comp_df["model"].tolist()
maes    = comp_df["MAE"].tolist()
colors  = ["#dc2626", "#f59e0b", "#16a34a"][:len(models)]
bars = axes[1].bar(range(len(models)), maes, color=colors, width=0.5, edgecolor="white")
axes[1].set_xticks(range(len(models)))
axes[1].set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
axes[1].set_ylabel("MAE (puan)")
axes[1].set_title("Model Karşılaştırması")
axes[1].set_ylim(0, max(maes) * 1.3)
for bar, val in zip(bars, maes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT_DIR / "comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  → Kaydedildi: outputs/step3/comparison_bar.png")
print(f"  → Kaydedildi: outputs/step3/comparison_table.csv")
print()
print("=" * 60)
print("Adım 3 Tamamlandı")
print("=" * 60)
print("""
Ne öğrendik:
  Fine-tuning persistence'ı geçti mi? (yukarıdaki tabloya bak)

  Eğer geçtiyse → "ILSA özelleştirmesi işe yarıyor" iddiamız doğru.
  Eğer geçmediyse → daha fazla eğitim verisi veya epoch gerekebilir,
                    ya da farklı kovaryatlar eklenmeli (Adım 4).

Sıradaki adım (Adım 4):
  Makale kanıtından gelen literatür ağırlıklarını kovaryat olarak ekle.
  Yani: "SES bu seride önemli, TIMSS ve PIRLS bunu gösteriyor" bilgisi
  modelin tahminini değiştiriyor mu?
""")
