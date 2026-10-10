# Yeniden üretilebilirlik notları

## Neler sağlanıyor?

- Dört temsil koşulunun okunabilir uygulaması
- Ana P0/P5 kampanyasının kilitli yapılandırma özeti
- 5×2 geliştirme protokolünün yapılandırma özeti
- P0/P5 ve P2 deneylerinde kullanılan dondurulmuş betikler
- Toplulaştırılmış geliştirme ve resmî test sonuçları
- Birim testleri

## Neler sağlanmıyor?

- Lisanslı ham veri ve gold etiketler
- Checkpoint dosyaları
- Örnek düzeyindeki tahminler
- İmzalı belgeler
- Model önbelleği veya Python sanal ortamı

## Dondurulmuş dosyaların doğrulanması

Aşağıdaki dosyalar deney kanıtından kopyalanmıştır. Depoya eklenmeden önce yerel SHA-256 değerleri ayrıca kaydedilir:

```powershell
Get-FileHash reproducibility/frozen/p0_p5/p0_p5_portable_v1.py -Algorithm SHA256
Get-FileHash reproducibility/frozen/p2_eq4/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4.py -Algorithm SHA256
Get-FileHash reproducibility/frozen/p2_eq4/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4.py -Algorithm SHA256
```

P0/P5 çalışma betiğinin beklenen SHA-256 değeri:

```text
441b455d6cb5f053c87d716380f10a7cec3670162581083df9ec51b9e84dcb4b
```

## Sınırlar

Dondurulmuş çalıştırıcılar belirli dizin sözleşmeleri, model önbelleği ve donanım kapıları içerir. Bunları doğrudan çalıştırmadan önce kaynak dosya yolları ile veri lisansını denetleyin. Dondurulmuş dosyaları değiştirmek yerine yeni denemeleri ayrı sürüm altında yürütün.

