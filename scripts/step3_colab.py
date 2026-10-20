# ============================================================
# ILSA TimesFM Fine-Tuning — Google Colab Versiyonu
# Çalıştırma ortamı: GPU (T4) — Runtime > Change runtime type > T4 GPU
#
# Hücre 1: Kurulum (bir kere çalıştır)
# ============================================================

# Hücre 1
# !pip install "timesfm[torch]" peft accelerate -q
# !pip install --upgrade torchao -q

# ============================================================
# Hücre 2: Veriyi Colab'a yükle
# Önce yerel makinende outputs/step1/panel_summary.csv değil,
# outputs/stage5/unified_panel.csv dosyasını yükle.
# Sol panelden klasör ikonuna tıkla → dosyayı sürükle bırak.
# Ya da aşağıdaki komutu çalıştır (Drive bağlantısı):
# ============================================================

# from google.colab import drive
# drive.mount('/content/drive')
# PANEL_PATH = "/content/drive/MyDrive/unified_panel.csv"

# Alternatif: direkt yükle
# from google.colab import files
# uploaded = files.upload()   # unified_panel.csv seç
# PANEL_PATH = "unified_panel.csv"

# ============================================================
# Hücre 3: Fine-tuning (buradan çalıştır)
# ============================================================

PANEL_PATH = "unified_panel.csv"   # yukarıdan hangisini seçtiysen

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import TimesFm2_5ModelForPrediction

# --- Sabitler ---
TEST_CYCLE   = 2022
MIN_CONTEXT  = 3
CONTEXT_LEN  = 5
HORIZON      = 1
NUM_WINDOWS  = 2000
EPOCHS       = 10
BATCH_SIZE   = 8     # T4 15GB — 231M model için 8 güvenli
LR           = 1e-4
LORA_R       = 4
LORA_ALPHA   = 8
SEED         = 42

np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cihaz: {device}")

# --- Veri ---
panel = pd.read_csv(PANEL_PATH)
TRAIN_SOURCES = [
    ("PISA","mathematics","PISA_mathematics"),
    ("PISA","reading","PISA_reading"),
    ("PISA","science","PISA_science"),
    ("TIMSS","mathematics","TIMSS_mathematics"),
    ("TIMSS","science","TIMSS_science"),
    ("PIRLS","reading","PIRLS_reading"),
]
all_series = []
for prog, dom, col in TRAIN_SOURCES:
    sub = panel[(panel["program"]==prog)&(panel["domain"]==dom)&(panel["cycle"]<TEST_CYCLE)].dropna(subset=[col])
    for _, grp in sub.groupby("country_iso3"):
        s = grp.sort_values("cycle")[col].tolist()
        if len(s) >= MIN_CONTEXT:
            all_series.append(s)
print(f"Toplam seri: {len(all_series)}")

windows = []
for _ in range(NUM_WINDOWS):
    s = all_series[np.random.randint(len(all_series))]
    if len(s) < 2:
        continue
    if len(s) < CONTEXT_LEN + HORIZON:
        cut = np.random.randint(1, len(s))
        ctx, tgt = s[:cut], s[cut:cut+HORIZON]
    else:
        st = np.random.randint(0, len(s)-CONTEXT_LEN-HORIZON+1)
        ctx, tgt = s[st:st+CONTEXT_LEN], s[st+CONTEXT_LEN:st+CONTEXT_LEN+HORIZON]
    if len(tgt) == HORIZON:
        windows.append((ctx, tgt))
print(f"Eğitim penceresi: {len(windows)}")

class ILSADataset(Dataset):
    def __init__(self, windows, ctx_len):
        self.windows = windows; self.ctx_len = ctx_len
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        ctx, tgt = self.windows[idx]
        padded = np.zeros(self.ctx_len, dtype=np.float32)
        arr = np.array(ctx, dtype=np.float32)
        padded[-len(arr):] = arr
        return {
            "past_values":   torch.tensor(padded, dtype=torch.float32),
            "future_values": torch.tensor(tgt,    dtype=torch.float32),
            "freq_token":    torch.tensor([0],     dtype=torch.long),
        }

dataloader = DataLoader(ILSADataset(windows, CONTEXT_LEN), batch_size=BATCH_SIZE, shuffle=True)

# --- Model + LoRA ---
print("Model yükleniyor...")
model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers", torch_dtype=torch.float32
)
lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                      target_modules="all-linear", lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Eğitilecek parametre: {trainable:,} ({100*trainable/sum(p.numel() for p in model.parameters()):.2f}%)")
base = model.base_model.model
base = base.to(device)

# --- Eğitim ---
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
epoch_losses = []
base.train()
for epoch in range(EPOCHS):
    losses = []
    for batch in dataloader:
        past   = batch["past_values"].to(device)
        future = batch["future_values"].to(device)
        freq   = batch["freq_token"].to(device)
        optimizer.zero_grad()
        out = base(past_values=past, future_values=future, freq_token=freq)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(out.loss.item())
    epoch_losses.append(np.mean(losses))
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss={epoch_losses[-1]:.4f}")

# --- Adapter kaydet ---
model.save_pretrained("ilsa-lora")
print("Adapter kaydedildi: ilsa-lora/")

# --- Test: PISA Matematik 2022 ---
print("\nDeğerlendirme: PISA Matematik 2022...")
sub = panel[(panel["program"]=="PISA")&(panel["domain"]=="mathematics")].dropna(subset=["PISA_mathematics"]).sort_values(["country_iso3","cycle"])
records = []
for country, grp in sub.groupby("country_iso3"):
    ctx_r = grp[grp["cycle"] < TEST_CYCLE]; tgt_r = grp[grp["cycle"] == TEST_CYCLE]
    if len(ctx_r) < MIN_CONTEXT or len(tgt_r) == 0: continue
    records.append({"country": country,
                    "ctx": ctx_r["PISA_mathematics"].tolist(),
                    "true": float(tgt_r["PISA_mathematics"].iloc[0]),
                    "persistence": float(ctx_r["PISA_mathematics"].iloc[-1])})

base.eval()
ft_preds = []
with torch.no_grad():
    for rec in records:
        ctx = np.array(rec["ctx"], dtype=np.float32)
        padded = np.zeros(CONTEXT_LEN, dtype=np.float32)
        padded[-min(len(ctx),CONTEXT_LEN):] = ctx[-min(len(ctx),CONTEXT_LEN):]
        out = base(past_values=torch.tensor(padded).unsqueeze(0).to(device),
                   freq_token=torch.tensor([[0]], dtype=torch.long).to(device))
        ft_preds.append(float(out.prediction_outputs[0,0].cpu()))

df = pd.DataFrame(records)
df["ft_pred"] = ft_preds
df["err_persist"] = abs(df["true"] - df["persistence"])
df["err_ft"]      = abs(df["true"] - df["ft_pred"])

print("\n========== SONUÇLAR ==========")
print(f"Persistence MAE     : {df['err_persist'].mean():.2f}")
print(f"Fine-tuned MAE      : {df['err_ft'].mean():.2f}")
delta = df['err_ft'].mean() - df['err_persist'].mean()
print(f"Fark                : {delta:+.2f}  ({'daha iyi ↓' if delta<0 else 'daha kötü ↑'})")
print(f"Fine-tuned kazanan  : {(df['err_ft']<df['err_persist']).sum()}/{len(df)} ülke")

df.to_csv("step3_results.csv", index=False)
print("\nstep3_results.csv kaydedildi.")
