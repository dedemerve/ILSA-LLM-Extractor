import os
import shutil
from pathlib import Path

print("="*100)
print("WORKSPACE CLEANUP ANALYSIS")
print("="*100)

base = Path("/Users/mrved/Desktop/ILSA_LLMs")
os.chdir(base)

# Test output klasörlerini tara
test_outputs = [
    d for d in base.iterdir() 
    if d.is_dir() and d.name.startswith('output_')
]

print(f"\n📁 Bulunan test output klasörleri: {len(test_outputs)}\n")

keep = []
delete = []

for folder in sorted(test_outputs):
    size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
    size_mb = size / (1024 * 1024)
    
    # Karar mantığı
    if 'diverse_10pdfs_v5_2' in folder.name:
        keep.append((folder, size_mb, "Final diverse test - 10 PDFs, 7 ILSAs"))
    elif 'FINAL' in folder.name:
        keep.append((folder, size_mb, "Marked as FINAL"))
    elif 'v5_2' in folder.name:
        keep.append((folder, size_mb, "v5.2 versiyonu"))
    else:
        delete.append((folder, size_mb, "Eski versiyon/intermediate"))

print("✅ KEEP (Saklanacak):")
for folder, size, reason in keep:
    print(f"   📁 {folder.name:50s} {size:6.1f} MB - {reason}")

print(f"\n❌ DELETE (Silinecek):")
total_free = 0
for folder, size, reason in delete:
    print(f"   📁 {folder.name:50s} {size:6.1f} MB - {reason}")
    total_free += size

print(f"\n💾 Toplam boşaltılacak alan: {total_free:.1f} MB")

print("\n" + "="*100)
print("ONAY BEKLENİYOR")
print("="*100)
print("\nBu klasörleri silmek için:")
print("   python workspace_cleanup.py --confirm")
print("\nÖnce gözden geçirin!")

import sys
if '--confirm' in sys.argv:
    print("\n🗑️  SİLME İŞLEMİ BAŞLIYOR...")
    for folder, size, reason in delete:
        print(f"   Siliniyor: {folder.name}")
        shutil.rmtree(folder)
    print(f"\n✅ {len(delete)} klasör silindi, {total_free:.1f} MB boşaltıldı")
