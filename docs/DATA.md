# Veri kapsamı

## Ana eğitim havuzu

P0 ve P5 ana kampanyasındaki 20 checkpointin tamamı aynı 9.550 kayıtlık eğitim havuzuyla eğitilmiştir. On eşleştirilmiş seed çifti aynı örnek sırasını ve sınıflandırıcı başlangıç sözleşmesini kullanır. Ana kampanyada ayrı bir doğrulama alt kümesi veya resmî test üzerinden checkpoint seçimi yapılmamıştır.

## Geliştirme protokolü

P0/P1/P2/P5 temsil ablation'ı, ana kampanyadan ayrı 1.500 kayıtlık bir havuz üzerinde 5×2 düzeninde yürütülmüştür. Her yönde 750 kayıt eğitim, kalan 750 kayıt değerlendirme için kullanılır. Bu 750+750 düzeni ana 9.550 kayıtlık eğitim kampanyasının bir parçası değildir.

## Resmî test

Resmî test 2.693 kayıt içerir. Eğitimde, hiperparametre belirlemede veya checkpoint seçiminde kullanılmamıştır. Checkpointler test değerlendirmesinden önce dondurulmuştur.

## Neden veri dosyaları yok?

SU-HS verisi erişim ve lisans koşullarına tabidir. Bu nedenle aşağıdaki içerikler depoya eklenmez:

- ham eğitim veya test metinleri,
- gold etiketler,
- örnek düzeyindeki tahminler,
- veri kullanım sözleşmeleri ve imzalı belgeler.

Depodaki sayısal sonuç dosyaları yalnızca toplulaştırılmış metrikleri içerir.

