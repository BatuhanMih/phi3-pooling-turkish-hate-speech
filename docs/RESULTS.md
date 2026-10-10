# Sonuçlar

## Ayrı 5×2 geliştirme protokolü

| Koşul | Ortalama Macro-F1 |
| --- | ---: |
| P0 | 0,5323 |
| P1 | 0,5822 |
| P2 | 0,5887 |
| P5 | 0,5891 |

Tekrar ortalamaları üzerinden farklar:

| Karşılaştırma | Ortalama fark | %95 Student-t GA |
| --- | ---: | ---: |
| P1−P0 | +0,0499 | [0,0370; 0,0628] |
| P2−P1 | +0,0065 | [−0,0106; 0,0235] |
| P5−P2 | +0,0004 | [−0,0013; 0,0020] |

Bu tablo, toplam gelişmenin büyük bölümünün son geçerli token yerine maskeli ortalama kullanılmasından geldiğini gösterir. Son dört katmanı birleştirmenin ek etkisi bu protokolde küçük ve belirsizdir.

## Resmî testte eşleştirilmiş yeniden değerlendirme

| Koşul | Ortalama Macro-F1 |
| --- | ---: |
| P0 | 0,7295 |
| P5 | 0,7106 |

P5−P0 farkı −0,0189'dur. 100.000 tekrarlı eşleştirilmiş hiyerarşik bootstrap %95 güven aralığı [−0,0312; −0,0067], kesin iki taraflı sign-flip p-değeri 0,0039'dur.

Bu değerlendirme araştırmacı tarafından yürütülmüş ve ara sonuçlarla örnek düzeyindeki etiketler kampanya bitene kadar kullanıcıya gösterilmemiştir. Bununla birlikte test kümesi daha önce kullanıldığı ve bağımsız veri sorumlusu bulunmadığı için sonuç bağımsız ya da dokunulmamış kör test olarak tanımlanmaz.

Makaledeki bilimsel sonuçlar için `results/` altındaki toplulaştırılmış JSON dosyaları kullanılabilir. `official_paired_analysis.json` eşleştirilmiş on seed'in sonuçlarını ve kilitli analiz ayarlarını içerir. Dondurulmuş P2 raporu `reproducibility/frozen/p2_eq4/` altındadır.

