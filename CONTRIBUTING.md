# Katkı rehberi

Değişiklikleri küçük ve denetlenebilir tutmak için şu akış kullanılır:

1. Ayrı bir dal açın.
2. Kod değişikliğiyle birlikte ilgili testi ekleyin veya güncelleyin.
3. Ham veri, checkpoint, tahmin dosyası ya da imzalı belge eklemediğinizi kontrol edin.
4. Testleri çalıştırın ve kısa bir açıklamayla pull request açın.

`reproducibility/frozen/` altındaki dosyalar tarihsel deney artifaktlarıdır. Bu dosyalar doğrudan değiştirilmez. Bir düzeltme gerekiyorsa yeni bir sürüm ve yeni SHA-256 kaydı eklenir.

Bilimsel sonucu etkileyen bir değişiklikte kullanılan veri bölümü, seed planı, model sürümü ve analiz yöntemi pull request açıklamasında açıkça belirtilmelidir.

