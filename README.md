# Phi-3 ile Türkçe Nefret Söylemi Tespiti

> Kaynak kısıtlı model uyarlamasında temsil havuzlama yöntemlerini karşılaştıran bir Türkçe NLP projesi.

---

## Genel bakış 🧭

Bu projede `microsoft/Phi-3-mini-4k-instruct` modeli, 4-bit QLoRA ile iki sınıflı Türkçe nefret söylemi tespitine uyarlanır. Amaç, son-token temsilinden maskeli ortalama havuzlamaya ve çoklu katman birleşimine geçişin etkisini ayrı koşullarda incelemektir.

Depoda küçük bir referans uygulama, deney yapılandırmaları, birim testleri, toplulaştırılmış sonuçlar ve deneylerde kullanılan dondurulmuş betikler bulunur. Lisanslı veri, test etiketleri ve model checkpointleri bulunmaz.

## Projenin kapsamı ✨

| Alan | Depoda bulunanlar |
| --- | --- |
| Temsil yöntemleri | P0, P1, P2 ve P5 için havuzlama bileşenleri |
| Model uyarlama | Phi-3 Mini, 4-bit NF4 QLoRA ve iki sınıflı sınıflandırıcı yapılandırması |
| Değerlendirme | Ayrı geliştirme ablation'ı ve eşleştirilmiş resmî test sonuçları |
| Yeniden üretilebilirlik | Yapılandırma özetleri, birim testleri, toplulaştırılmış metrikler ve dondurulmuş betikler |
| Veri paylaşımı | Lisanslı ham veri, test etiketleri ve checkpointler yayımlanmaz |

## Karşılaştırılan yöntemler 🔬

| Koşul | Temsil yöntemi |
| --- | --- |
| P0 | Son gizli katmanın son geçerli token temsili |
| P1 | Son gizli katmana maskeli ortalama havuzlama |
| P2 | Son dört gizli katmanın eşit ağırlıklı ortalaması ve maskeli ortalama havuzlama |
| P5 | Son dört gizli katmanın öğrenilebilir softmax birleşimi ve maskeli ortalama havuzlama |

Bu ayrım, token havuzlama ile katman birleştirmenin katkılarını birbirinden ayırır. Uygulama ayrıntıları [mimari notlarında](docs/ARCHITECTURE.md) açıklanır.

## Teknoloji ve yapılandırma 🛠️

| Bileşen | Kullanılan yaklaşım |
| --- | --- |
| Temel model | Phi-3 Mini 4K Instruct |
| Uyarlama | 4-bit NF4 QLoRA |
| LoRA hedefi | `qkv_proj` |
| Azami dizi uzunluğu | 128 token |
| Sınıflandırma | Bias içermeyen iki sınıflı doğrusal katman |
| Ana kampanya | 10 eşleştirilmiş seed için 10 P0 + 10 P5 checkpoint |
| Değerlendirme | Macro-F1, ayrı geliştirme ve resmî test protokolleri |

Tam ayarlar ve protokol sınırları [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), [`docs/DATA.md`](docs/DATA.md) ve `configs/` altında açıklanır.

## Kod mimarisi 🏗️

```text
Phi-3 gizli durumları + attention mask
                    │
                    ▼
     P0 / P1 / P2 / P5 havuzlama
                    │
                    ▼
        metin düzeyinde temsil
                    │
                    ▼
    iki sınıflı doğrusal sınıflandırıcı
```

`PoolingHead` dört yöntemi ortak bir arayüzle sunar. Referans modül havuzlama mantığını gösterir; tam Phi-3 eğitim hattının veya lisanslı veri kümesinin yerini almaz.

### Depo düzeni 📁

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

## Başlangıç 🚀

Python 3.11 önerilir. Depoyu klonlayıp sanal ortam oluşturun:

```bash
git clone https://github.com/BatuhanMih/phi3-pooling-turkish-hate-speech.git
cd phi3-pooling-turkish-hate-speech
python -m venv .venv
```

Windows PowerShell'de `.venv\Scripts\Activate.ps1`, macOS/Linux'ta `source .venv/bin/activate` ile ortamı etkinleştirin. PyTorch'u donanımınıza uygun seçenekle [resmî kurulum sayfasından](https://pytorch.org/get-started/locally/) kurun. Sonra:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

Havuzlama modülünü yapay gizli durum tensörleriyle deneyebilirsiniz; bu örnek model eğitimi veya veri indirmesi başlatmaz:

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

## Deneyler ve sonuçlar 📊

**Geliştirme ablation'ı:** Ayrı 1.500 kayıtlık 5×2 protokolünde her yönde 750 kayıt eğitim, 750 kayıt değerlendirme için kullanılmıştır.

| Koşul | Ortalama Macro-F1 |
| --- | ---: |
| P0 | 0,5323 |
| P1 | 0,5822 |
| P2 | 0,5887 |
| P5 | 0,5891 |

Bu protokolde en büyük artış, son-token temsilinden maskeli ortalamaya geçiştedir. P2 ve P5'in ek kazanımları daha küçüktür.

**Resmî testte eşleştirilmiş yeniden değerlendirme:** Önceden dondurulmuş 10 P0 ve 10 P5 checkpointi, aynı 2.693 test kaydı üzerinde değerlendirilmiştir. Bu checkpointler ayrı 9.550 kayıtlık eğitim havuzunda eğitilmiştir; geliştirmedeki 750+750 katları ana kampanyanın eğitim kümesi değildir.

| Koşul | Ortalama Macro-F1 |
| --- | ---: |
| P0 | 0,7295 |
| P5 | 0,7106 |

Eşleştirilmiş P5−P0 farkı −0,0189; %95 güven aralığı [−0,0312; −0,0067] ve kesin iki taraflı sign-flip p-değeri 0,0039'dur. Geliştirme ile resmî test farklı protokollerdir; sonuçları tek bir sıralama gibi birleştirmemek gerekir. Test araştırmacı tarafından yürütülmüştür; **bağımsız kör değerlendirme olarak sunulmaz**.

Yöntem, istatistikler ve sınırlılıklar için [`docs/RESULTS.md`](docs/RESULTS.md) dosyasına bakın.

## Veri ve erişim 🔒

SU-HS ham verisi, resmî test dosyası, gold etiketler, örnek düzeyindeki tahminler, checkpointler ve imzalı kullanım belgeleri depoda yoktur. Veri kullanımı için sağlayıcının erişim ve lisans koşulları ayrıca karşılanmalıdır. Ayrıntılar [`docs/DATA.md`](docs/DATA.md) içinde yer alır.

## Proje ekibi 👥

- [Batuhan Mıh](https://github.com/BatuhanMih)
- [Mustafa Kurtulan](https://github.com/MustafaKurtulan)
- [Faruk Topal](https://github.com/faruktopal)

Ekip bilgileri [`CONTRIBUTORS.md`](CONTRIBUTORS.md) dosyasında da bulunur. GitHub'daki ekip listesi, kişilerin katkı commit'leri ve davet kabul durumuyla aynı şey değildir.

## Katkı, atıf ve lisans 📄

Katkı akışı için [`CONTRIBUTING.md`](CONTRIBUTING.md), yazılım atfı için [`CITATION.cff`](CITATION.cff) dosyasına bakın. Depodaki özgün kod MIT lisanslıdır; veri ve üçüncü taraf model bileşenleri kendi lisanslarına tabidir. Ayrıntılar [`LICENSE`](LICENSE) dosyasındadır.
