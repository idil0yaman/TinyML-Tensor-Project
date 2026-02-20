# TinyML Tensor Sistemi

Bu proje, C dilinde basit bir **quantization + ReLU** tabanlı TinyML tensor sistemi örneğidir.  
Amaç: Negatif değerlerin int8 quantization ile bellekte saklanması ve ReLU aktivasyonu sonrası sıfırlanmasını göstermek.

## Özellikler
- Float32 → Int8 quantization (negatifleri korur)
- De-quantization ile geri okuma
- ReLU aktivasyonu
- Basit inference fonksiyonu
