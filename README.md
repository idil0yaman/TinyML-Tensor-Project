# TinyML-Tensor-Project
Mikrodenetleyiciler (Arduino/ESP32) için C dilinde geliştirilmiş, Union ve Int8 Quantization tabanlı bellek dostu TinyML Tensör ve Çıkarım Motoru.

# 🧠 TinyML Dynamic Tensor & AI Inference Engine

Bu proje, RAM kapasitesi kısıtlı olan gömülü sistemlerde (ESP32, Arduino, STM32 vb.) yapay zeka modellerinin verimli bir şekilde çalıştırılabilmesi için tasarlanmış, C tabanlı dinamik bir tensör kütüphanesi ve çıkarım (inference) motorudur.

## 🎯 Proje Amacı ve Çözüm Yaklaşımı
Standart yapay zeka modellerinde kullanılan 32-bit Float veri tipleri, mikrodenetleyicilerin belleğini hızla tüketir. Bu projede, standart veri yapıları yerine düşük seviyeli C özellikleri kullanılarak sistem optimize edilmiştir:

1. **Bellek Optimizasyonu (`union`):** Farklı veri tiplerinin aynı bellek adresini paylaşması sağlanarak RAM kullanımı minimize edilmiştir.
2. **Quantization (Nicemleme):** Float32 verileri, dinamik Scale ve Zero-Point hesaplamalarıyla 8-bit Integer (Int8) formatına sıkıştırılmış ve bellek ayak izi %75 oranında azaltılmıştır.
3. **Yapay Zeka Çıkarımı:** Sıkıştırılmış veriler (Quantized Data) ve model ağırlıkları kullanılarak matris çarpımı (Inference) gerçekleştirilmiştir.
4. **Non-Linearity (ReLU):** Yapay zeka modelinin negatif değerleri filtrelemesi için ReLU aktivasyon fonksiyonu sisteme entegre edilmiştir.

## 🛠 Kullanılan Teknolojiler ve Mimari
* **Programlama Dili:** C (Standart C99)
* **Veri Yönetimi:** `struct`, `union`, `enum`
* **Geliştirme Ortamı:** CLion / GCC
* **Geliştirme Metodolojisi:** Agentic AI Coding

## 🤖 Agentic Kodlama Süreci
Bu proje geliştirilirken yapay zeka (LLM) araçları bir "Thought Partner" (Düşünce Ortağı) olarak kullanılmış ve **Agentic Kodlama** prensipleri uygulanmıştır:
* **Tasarım:** Mikrodenetleyiciler için float ve int8 tutabilen hibrit bir yapı tasarımının temelleri AI yardımıyla atıldı.
* **Giriş Güvenliği (Input Validation):** Kullanıcının hatalı veri (harf) girmesi durumunda sistemin çökmesini engelleyen `clear_buffer()` mantığı iteratif olarak geliştirildi.
* **Kapsam Genişletme:** Sistemin sadece bir veri deposu olmaktan çıkıp, `apply_relu` ve `perform_inference` fonksiyonlarıyla çalışan bir TinyML motoruna dönüşmesi sağlandı.
