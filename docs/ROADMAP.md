# Yol Haritası (Roadmap)

Hedef: RETECO Track 1 (TEMPO) 1a + 1b için resmi BM25'i (dev 0.115 / 0.118) belirgin şekilde
geçen, ablation'larla desteklenmiş bir sistem ve ACL formatında makale.

| Faz | İçerik | Çıktı | Durum |
|---|---|---|---|
| 0 | Görev analizi, veri şeması, proje iskeleti, makale şablonu | bu repo | ✅ |
| 1 | Tüm Track 1 verisi (v1.1), pasaj BM25 (resmî baseline'ın üstünde: .197 / .223) | `bench_dev.json` | ✅ |
| 2 | Metadata / zamansal veri analizi (altın dokümanlar vs. BM25 hataları) | makale Tablo 1 | ✅ |
| 3 | Zamansal prior, yıl indeksi, zamansal KG, zamansal PRF (train'de ayar) | makale Tablo 3 | ✅ (fayda yok) |
| 4 | Dense indeks (Qwen3-Embedding-0.6B, A100) + RAPTOR / temporal RAPTOR | makale Tablo 3 | ✅ |
| 5 | LLM sorgu planlayıcı (Qwen3-8B, olay örgüsü, multi-query, q2d, LLM-PRF) | makale Tablo 3 | ✅ |
| 6 | Cross-encoder vs. cross-decoder rerank (top-100 → top-10), havuz seçimi | makale Tablo 4 | ✅ |
| 7 | Nihai sistem v1: 1a .371, 1b .341 (dev) | `bench_dev_v1.json` | ✅ |
| 8 | Domain ve intent analizi, figürler, makale ve sunum | `paper/main.pdf`, sunum | ✅ |
| 9 | Küçük LLM (Qwen3-1.7B) ile 1,17M belge + sorgu metadata'sı; dönem filtresi / prior / metadata indeksi | makale Tablo 5 | ✅ (dönem: fayda yok) |
| 10 | Komşu olaylar + alt sorgular (Qwen3-8B), ReasonIR-8B (± LLM akıl yürütmesi) | makale Tablo 2, 5 | ✅ |
| 11 | Nihai sistem v2: 1a .387, 1b .371 (dev); TEMPO tablosunda 1.730 soruda 33.4 | makale Tablo 3 | ✅ |
| 12 | 4B reranker'ı 1b'de çalıştırmak, listwise; Ocak 2027 test penceresinde submission | runs | ⏳ |

## Kurallar
- Ayarlama **train**'de, raporlama **dev**'de. Guidance/gold hiçbir zaman değerlendirilen sorgu için girdi değildir.
- Her deney bir YAML config'tir; sonuç JSON'ları `results/metrics/` altına düşer, makale tabloları buradan üretilir.
- LLM çağrıları `.cache/llm/` altında önbelleklenir (tekrar üretilebilirlik + maliyet).
