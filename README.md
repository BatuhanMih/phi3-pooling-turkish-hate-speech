# Phi-3 ile Türkçe Nefret Söylemi Tespiti

> Phi-3 Mini üzerinde farklı temsil havuzlama yöntemlerini karşılaştıran, kaynak kısıtlı Türkçe metin sınıflandırma projesi.

## Genel bakış

Bu projede `microsoft/Phi-3-mini-4k-instruct` modeli, 4-bit QLoRA ile iki sınıflı Türkçe nefret söylemi tespitine uyarlanır. Amaç, son-token temsilinden maskeli ortalama havuzlamaya ve çoklu katman birleşimine geçişin etkisini ayrı koşullarda incelemektir.

Depoda küçük bir referans uygulama, deney yapılandırmaları, birim testleri, toplulaştırılmış sonuçlar ve deneylerde kullanılan dondurulmuş betikler bulunur. Lisanslı veri, test etiketleri ve model checkpointleri bulunmaz.

## Karşılaştırılan yöntemler

| Koşul | Temsil yöntemi |
| --- | --- |
| P0 | Son gizli katmanın son geçerli token temsili |
| P1 | Son gizli katmana maskeli ortalama havuzlama |
| P2 | Son dört gizli katmanın eşit ağırlıklı ortalaması ve maskeli ortalama havuzlama |
| P5 | Son dört gizli katmanın öğrenilebilir softmax birleşimi ve maskeli ortalama havuzlama |

Bu ayrım, token havuzlama ile katman birleştirmenin katkılarını birbirinden ayırır.

## Teknoloji ve yapılandırma

| Bileşen | Kullanılan yaklaşım |
| --- | --- |
| Temel model | Phi-3 Mini 4K Instruct |
| Uyarlama | 4-bit NF4 QLoRA |
| LoRA hedefi | `qkv_proj` |
| Azami dizi uzunluğu | 128 token |
| Sınıflandırma | Bias içermeyen iki sınıflı doğrusal katman |
| Değerlendirme | Macro-F1, ayrı geliştirme ve resmî test protokolleri |

Tam ayarlar ve protokol sınırları [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), [`docs/DATA.md`](docs/DATA.md) ve `configs/` altında açıklanır.

## Kod mimarisi

```text
src/phi3_pooling/          P0/P1/P2/P5 için okunabilir havuzlama bileşenleri
configs/                  Deney yapılandırmalarının özetleri
scripts/                  Sonuç özetleme araçları
tests/                    Birim testleri
docs/                     Mimari, veri, deney ve yeniden üretilebilirlik notları
results/                  Toplulaştırılmış metrikler
reproducibility/frozen/   Deneylerde kullanılan, değiştirilmeden saklanan betikler
```

`src/phi3_pooling/` yöntemleri anlamak ve yeni uygulamalara başlangıç yapmak içindir. `reproducibility/frozen/` altındaki betikler ise yürütülmüş deneylerin kayıtlarıdır; taşınabilir, tek komutluk eğitim paketi olarak değerlendirilmemelidir.

## Kurulum ve hızlı örnek

Python 3.11 önerilir. PyTorch'u donanımınıza uygun seçenekle [resmî kurulum sayfasından](https://pytorch.org/get-started/locally/) kurduktan sonra:

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

Havuzlama modülünü sahte gizli durum tensörleriyle deneyebilirsiniz; bu örnek model eğitimi veya veri indirmesi başlatmaz:

```python
import torch
from phi3_pooling import PoolingHead

hidden_states = tuple(torch.randn(2, 8, 32) for _ in range(4))
attention_mask = torch.tensor([
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
])

representations = PoolingHead(mode="p2")(hidden_states, attention_mask)
print(representations.shape)
```

## Sonuçlara kısa bakış

Ayrı 5×2 geliştirme protokolünde ortalama Macro-F1 değerleri P0/P1/P2/P5 için sırasıyla 0,5323 / 0,5822 / 0,5887 / 0,5891'dir. Bu protokoldeki en büyük artış, son-token temsilinden maskeli ortalamaya geçişte görülür.

2.693 kayıtlık resmî testte, aynı kayıtlar üzerinde eşleştirilmiş P0/P5 yeniden değerlendirmesinde ortalama Macro-F1 değerleri sırasıyla 0,7295 ve 0,7106'dır. Geliştirme ve resmî test farklı protokollerdir; sonuçları tek bir sıralama gibi birleştirmemek gerekir. Test değerlendirmesi araştırmacı tarafından yürütülmüştür; bağımsız kör değerlendirme olarak sunulmaz.

Yöntem, istatistikler ve sınırlılıklar için [`docs/RESULTS.md`](docs/RESULTS.md) dosyasına bakın.

## Veri ve erişim

SU-HS ham verisi, resmî test dosyası, gold etiketler, örnek düzeyindeki tahminler, checkpointler ve imzalı kullanım belgeleri depoda yoktur. Veri kullanımı için sağlayıcının erişim ve lisans koşulları ayrıca karşılanmalıdır. Ayrıntılar [`docs/DATA.md`](docs/DATA.md) içinde yer alır.

## Proje ekibi

- [Batuhan Mıh](https://github.com/BatuhanMih)
- [Mustafa Kurtulan](https://github.com/MustafaKurtulan)
- [Faruk Topal](https://github.com/faruktopal)

Ekip bilgileri [`CONTRIBUTORS.md`](CONTRIBUTORS.md) dosyasında da bulunur. GitHub'daki ekip listesi, kişilerin katkı commit'leri ve davet kabul durumuyla aynı şey değildir.

## Katkı ve lisans

Katkı akışı için [`CONTRIBUTING.md`](CONTRIBUTING.md) dosyasına bakın. Depodaki özgün kod MIT lisanslıdır; veri ve üçüncü taraf model bileşenleri kendi lisanslarına tabidir.
