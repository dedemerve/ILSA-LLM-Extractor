# Faz 3 — RQ2 için önceden kaydedilmiş karar kuralı

**Yazılma zamanı:** 27 Ekim 2026; Faz 3 analizleri (placebo, permütasyon, veri güdümlü ceza, özellik seçimi) çalıştırılmadan önce.
**Kilit:** Bu dosyanın sha256 değeri, analiz başlamadan önce `outputs/rq2_v3/manifest.json` içine `preregistration_sha256` olarak yazılır. Analiz scripti, dosya değişmişse çalışmayı reddeder.

**Şeffaflık notu.** Faz 2'nin betimsel sonuçları (M0A, M0 ve M1'in nokta MAE değerleri) bu kural yazılmadan önce görüldü. Placebo, permütasyon, veri güdümlü ceza ve KB'ye dayalı özellik seçimi sonuçları ise görülmedi. Kural, prompttaki 3.5 maddesinin metnine bağlı kalınarak yazıldı; Faz 2 sonuçlarına göre değiştirilmedi.

## Kapsam

- **Birim:** Ana program-alanlar (8): PISA matematik, okuma ve fen; TIMSS G8 matematik ve fen; TIMSS G4 matematik ve fen; PIRLS okuma.
- **Örneklem:** `run_unified_backtest.py` ile aynı tam fold'lar ve aynı test satırları.
  - **Ana analiz:** 2025 test döngüsü hariç.
  - **Duyarlılık analizi:** 2025 dahil. Karar yalnızca ana analize göre verilir.
- **Ek programlar (ICCS, ICILS, PIAAC):** Kurala dahil değil; tek fold oldukları için yalnızca betimsel.
- **Metrik:** Program-alan içinde tüm test satırları üzerinden havuzlanmış MAE (puan).

## Karşılaştırılan modeller (hepsi aynı Ridge, aynı iç zaman sıralı alpha seçimi)

| Ad | Kovaryat ağırlığı |
|---|---|
| M0A | Kovaryat yok (yalnızca skor özellikleri) |
| M0 | Tüm kovaryatlar, ağırlık 1 |
| M1 | Kovaryatlar, KB v3 ağırlığı w_j ile, ölçek sqrt(w_j) |
| Uniform_c | Tüm kovaryatlar için w = c, c ∈ {0.75, 0.5, 0.25} |
| Permütasyon null | 200 çekiliş. Her çekilişte programın her kovaryatına, v3'teki 12 pozitif ağırlıktan iadesiz rastgele bir değer atanır; atama program-alan içinde tüm fold'larda sabittir (seed 20261027). Tek kovaryatlı programlarda da null dağılımının dejenere olmaması için bu tanım seçildi |
| DataDriven_block | Ortak kovaryat çarpanı c ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}; alpha ile birlikte iç zaman sıralı doğrulamayla seçilir |
| DataDriven_adaptive | Eğitim verisinde M0'ın standartlaştırılmış katsayılarından w_j = abs(β_j) / max abs(β), alt sınır 0.01; ardından sqrt(w_j) ölçekli Ridge (adaptive ridge). Yalnızca eğitim verisi kullanılır |
| KB_top_k | Programın kovaryatlarından KB ağırlığına göre ilk k tanesi, k ∈ {1, 2, 3} (kovaryat sayısını aşmayacak şekilde); her fold'da eğitim sinyali olan kovaryatlar arasında |
| Random_top_k null | 200 çekiliş. Kovaryatların rastgele sıralanmasıyla aynı top-k kuralı uygulanır |

## Karar kuralı (her program-alan için, ana analiz)

KB'nin o program-alanda **tahmini iyileştirdiği** ancak aşağıdaki üç koşulun **hepsi** sağlanırsa söylenir:

1. **M1, M0A'dan anlamlı ve pratik olarak daha iyi:** ΔMAE(M0A − M1) ≥ 1.0 puan, ve bu farkın ülke kümeli bootstrap %95 güven aralığının alt sınırı > 0 (Faz 4; 2 000+ tekrar, Holm düzeltmesi 8 program-alan üzerinden).
2. **KB'nin içeriği önemli:** MAE(M1), permütasyon null dağılımının 5. yüzdeliğinden küçük; yani M1 rastgele ağırlık atamalarının en az %95'inden iyi.
3. **M1, M0'dan daha iyi:** ΔMAE(M0 − M1) > 0.

**Genel RQ2 iddiası:**
- "KB tahminleri tutarlı biçimde iyileştiriyor" ifadesi ancak kural 8 program-alanın en az 6'sında sağlanırsa kullanılır.
- Kural 1–5 program-alanda sağlanırsa: "yalnızca şu program-alanlarda iyileştirme".
- Hiçbirinde sağlanmazsa: "KB entegrasyonunun tahmine katkısı olduğuna dair kanıt yok".
- Faz 4'teki TOST, M1 − M0A farkının ±1 puan içinde olduğunu gösterirse, "pratik olarak eşdeğer" ifadesi eklenir.

**İkincil yol (özellik seçimi):** KB_top_k'nin MAE'si Random_top_k null dağılımının 5. yüzdeliğinden küçükse ve M0A'dan en az 1 puan iyiyse, "KB'ye dayalı özellik seçimi yardımcı oluyor" denir. Aynı Holm düzeltmesi uygulanır. Bu yol birincil kararı değiştirmez; ayrıca raporlanır.

**Placebo yorumu (betimsel):** Uniform_c veya DataDriven modellerinden biri M1'den iyiyse, M1'in M0'a karşı herhangi bir kazancı KB içeriğine değil, genel kovaryat küçültmesine atfedilir.

## Değiştirilemez ayrıntılar

- Bootstrap: ülke kümeli; program-alan içinde ülkeler yeniden örneklenir ve o ülkenin tüm fold'lardaki satırları birlikte alınır; seed 20261027.
- Çoklu karşılaştırma: Kural 1'in p-değerleri için Holm (m = 8).
- Eşdeğerlik marjı: ±1.0 puan. Gerekçe Faz 4'te ülke ortalamalarının medyan SE'si (yaklaşık 2.5) ve yayımlanmış link error değerleriyle belgelenecek; marj değişmez.
- Çalıştırma sonrası kural değiştirilirse, değişiklik ve gerekçesi bu dosyaya ek olarak yazılır; orijinal metin silinmez.
