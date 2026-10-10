# Phi-3 ile Türkçe nefret söylemi sınıflandırması

Bu depo, Phi-3-mini-4k-instruct modelinde temsil havuzlama seçiminin Türkçe nefret söylemi sınıflandırmasına etkisini inceleyen çalışmanın kod ve yeniden üretilebilirlik dosyalarını içerir.

Çalışmanın temel karşılaştırması dört koşuldan oluşur:

| Koşul | Temsil |
| --- | --- |
| P0 | Son gizli katmandaki son geçerli token |
| P1 | Son gizli katmana maskeli ortalama |
| P2 | Son dört katmanın eşit ağırlıklı ortalaması, ardından maskeli ortalama |
| P5 | Son dört katmanın öğrenilebilir softmax ağırlıklarıyla birleşimi, ardından maskeli ortalama |

Kod iki bölüm halinde tutulur. `src/phi3_pooling/` küçük ve okunabilir bir referans uygulamadır. `reproducibility/frozen/` ise deneylerde kullanılan dondurulmuş betikleri içerir; bu dosyalar tarihsel kanıt olarak değiştirilmeden saklanır.

## Sonuçların kısa özeti

1.500 kayıtlık ayrı 5×2 geliştirme protokolünde ortalama Macro-F1 değerleri P0 için 0,5323; P1 için 0,5822; P2 için 0,5887 ve P5 için 0,5891'dir. Kazancın büyük bölümü son-token temsilinden maskeli ortalamaya geçişte ortaya çıkmıştır.

2.693 kayıtlık resmî test üzerinde yapılan eşleştirilmiş yeniden değerlendirmede P0 ortalaması 0,7295, P5 ortalaması 0,7106'dır. Eşleştirilmiş fark P5−P0 = −0,0189; %95 güven aralığı [−0,0312, −0,0067] ve kesin iki taraflı sign-flip p-değeri 0,0039'dur. Bu değerlendirme araştırmacı tarafından yürütülmüş, etiketler sonuç tamamlanana kadar gizlenmiştir; bağımsız kör test olarak sunulmaz.

Ayrıntılı sayılar ve protokol ayrımı için [`docs/RESULTS.md`](docs/RESULTS.md) dosyasına bakın.

## Kurulum

Python 3.11 önerilir. PyTorch kurulumu CUDA sürümüne bağlı olduğu için önce uygun paketi [PyTorch kurulum sayfasından](https://pytorch.org/get-started/locally/) kurun. Ardından:

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

Hızlı örnek:

```python
import torch
from phi3_pooling import PoolingHead

hidden_states = tuple(torch.randn(2, 8, 32) for _ in range(4))
attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 0, 0]])

head = PoolingHead(mode="p2")
representations = head(hidden_states, attention_mask)
```

## Depo düzeni

```text
configs/                    Deney sözleşmelerinin okunabilir özetleri
docs/                       Veri, mimari, deney ve sonuç notları
results/                    Yalnızca toplulaştırılmış sonuçlar
scripts/                    Sonuç dosyalarını özetleyen yardımcı araçlar
src/phi3_pooling/           Okunabilir referans uygulama
tests/                      Birim testleri
reproducibility/frozen/     Deneylerde kullanılan dondurulmuş betikler
```

## Veri ve özel dosyalar

SU-HS ham verisi, resmî test dosyası, gold etiketler, örnek düzeyindeki tahminler, checkpointler ve imzalı lisans belgeleri bu depoda yer almaz. Veriyi kullanmak isteyen araştırmacıların veri sağlayıcısının lisans ve erişim koşullarını ayrıca karşılaması gerekir.

## Katkı sahipleri

- Batuhan Mıh — araştırma, deney tasarımı, modelleme ve makale
- Mustafa Kurtulan — proje katkıcısı
- Faruk Topal — proje katkıcısı

Bu bölüm proje katkılarını gösterir. Makale yazarlığı ve akademik atıf bilgisi [`CITATION.cff`](CITATION.cff) dosyasında ayrıca tanımlanmıştır.

## Lisans

Bu depodaki özgün kod MIT lisansıyla yayımlanır. Veri kümeleri, model ağırlıkları ve üçüncü taraf bileşenler kendi lisanslarına tabidir. Ayrıntılar için [`LICENSE`](LICENSE) ve [`docs/DATA.md`](docs/DATA.md) dosyalarına bakın.

