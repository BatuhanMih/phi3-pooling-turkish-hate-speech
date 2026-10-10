# Kod mimarisi

Bu depo, deney kodunu tek bir büyük çalıştırıcı içinde açıklamaya çalışmak yerine iki ayrı ihtiyacı ayırır.

## Okunabilir uygulama

`src/phi3_pooling/pooling.py` dört koşulun temsil mantığını küçük fonksiyonlarla gösterir:

- `last_token_pool`: P0
- `masked_mean_pool`: P1'in token havuzlaması
- `equal_layer_mix` + `masked_mean_pool`: P2
- `LearnedLayerMix` + `masked_mean_pool`: P5

`PoolingHead`, bu parçaları ortak bir çağrı arayüzü altında birleştirir. P5'in dört katman ağırlığı sıfır logitlerle başlar; softmax başlangıçta her katmana 0,25 ağırlık verir ve ağırlıklar eğitim sırasında öğrenilir.

Sınıflandırıcı bu modülün dışında tutulur. Çalışmada iki sınıflı, bias içermeyen doğrusal sınıflandırıcı kullanılmıştır.

## Dondurulmuş çalışma betikleri

`reproducibility/frozen/` altındaki dosyalar deney sırasında kullanılan çalışma kodlarıdır. Okunabilirlik amacıyla yeniden düzenlenmemiştir. Bu dosyalardaki sürüm adları, güvenlik kapıları ve ayrıntılı kayıt üretimi tarihsel deney sözleşmesinin parçasıdır.

Dondurulmuş betikler taşınabilir bir “tek komutla yeniden üretim” paketi değildir. Veri erişimi, donanım, model önbelleği ve lisans koşulları ayrıca sağlanmalıdır. Yeni çalışmalar için `src/` altındaki modüler uygulamayı başlangıç noktası olarak kullanmak daha uygundur.

