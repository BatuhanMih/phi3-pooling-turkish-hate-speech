# Deneyler

## Model ve uyarlama

Temel model `microsoft/Phi-3-mini-4k-instruct` sürüm `f39ac1d28e925b323eae81227eaba4464caced4e` olarak kilitlenmiştir. Azami dizi uzunluğu 128'dir. Uyarlama 4-bit NF4 QLoRA ile yapılmış; LoRA yalnızca `qkv_proj` modülüne, rank 16, alpha 16 ve dropout 0,0697368 değerleriyle uygulanmıştır.

Ana P0/P5 kampanyasında her checkpoint üç epoch ve 7.164 optimizer adımıyla eğitilmiştir. Per-device batch 1, gradient accumulation 4 ve etkin batch 4'tür. Öğrenme oranı 8,687484×10⁻⁵; warmup oranı 0,195124; weight decay 0,004709 ve label smoothing 0,05'tir. Optimizasyon `paged_adamw_32bit` ile yapılmış, gradient checkpointing açık tutulmuştur.

Model girişinin sonuna yeni bir terminal token eklenmez. Tahmin `argmax` ile alınır; eşik ayarı ve kalibrasyon yapılmaz.

## Donanım

Ana 20-checkpoint kampanyası, 6.141 MiB VRAM bulunan NVIDIA GeForce RTX 4050 Laptop GPU üzerinde yürütülmüştür. Ayrı geliştirme protokolü için kanıt paketi aynı ayrıntıda doğrulanmış cihaz ataması sağlamadığından bu protokole donanım modeli yazılmamıştır.

## Protokollerin ayrılması

Geliştirme ablation'ı ile resmî test değerlendirmesi aynı veri üzerinde yapılmamıştır. Geliştirme sonuçları temsil mekanizmasını parçalarına ayırır; resmî test ise önceden dondurulmuş P0/P5 checkpointlerini aynı test kayıtlarında eşleştirilmiş biçimde karşılaştırır. İki sonuç grubu doğrudan tek bir sıralama tablosu gibi yorumlanmamalıdır.

