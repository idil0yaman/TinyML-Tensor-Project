#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

// --- VERI YAPILARI ---
typedef enum { TYPE_FLOAT32 = 0, TYPE_INT8_QUAN = 1 } TensorType;
typedef union { float* f32_ptr; int8_t* i8_ptr; } TensorData;

typedef struct {
    TensorType type;
    TensorData data;
    int length;
    float scale;
    int8_t zero_point;
} TinyTensor;

// --- YARDIMCI VE CEKIRDEK FONKSIYONLAR ---
void clear_buffer() { int c; while ((c = getchar()) != '\n' && c != EOF); }

// 1. Quantization (Sıkıştırma) - Negatifleri de kapsar
int quantize_tensor(TinyTensor* tensor, float* input, int len) {
    float max_val = input[0], min_val = input[0];
    for(int i = 1; i < len; i++) {
        if(input[i] > max_val) max_val = input[i];
        if(input[i] < min_val) min_val = input[i];
    }

    // Scale hesaplarken 0'a bölme hatasını önle
    if (max_val == min_val) {
        tensor->scale = 1.0f;
    } else {
        tensor->scale = (max_val - min_val) / 255.0f;
    }

    tensor->zero_point = (int8_t)roundf(-min_val / tensor->scale - 128);
    tensor->length = len;
    tensor->type = TYPE_INT8_QUAN;

    tensor->data.i8_ptr = (int8_t*)malloc(len * sizeof(int8_t));
    if (tensor->data.i8_ptr == NULL) {
        printf("HATA: Bellek yetersiz!\n");
        return 0;
    }

    for (int i = 0; i < len; i++) {
        // Formül: q = (f / scale) + zero_point
        int32_t q = (int32_t)roundf(input[i] / tensor->scale) + tensor->zero_point;

        // Clamping: -128 ile 127 arasına sıkıştır (Negatifler burada korunur!)
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        tensor->data.i8_ptr[i] = (int8_t)q;
    }
    return 1;
}

// 2. De-Quantization
float get_tensor_value(TinyTensor* tensor, int index) {
    return (tensor->type == TYPE_INT8_QUAN) ? (float)(tensor->data.i8_ptr[index] - tensor->zero_point) * tensor->scale : 0.0f;
}

// 3. ReLU Aktivasyonu (Negatifleri Sıfırlar)
void apply_relu(TinyTensor* tensor) {
    if (tensor->type != TYPE_INT8_QUAN) return;
    for (int i = 0; i < tensor->length; i++) {
        // Eğer değer Zero Point'ten küçükse (yani gerçekte negatifse)
        if (tensor->data.i8_ptr[i] < tensor->zero_point) {
            tensor->data.i8_ptr[i] = tensor->zero_point; // Sıfıra eşitle
        }
    }
}

// 4. Inference
float perform_inference(TinyTensor* input, TinyTensor* weights, float bias) {
    float sum = 0.0f;
    for (int i = 0; i < input->length; i++) {
        sum += get_tensor_value(input, i) * get_tensor_value(weights, i);
    }
    return sum + bias;
}

// --- ANA PROGRAM ---
int main() {
    int n = 3;
    float trained_weights[] = {0.5f, -0.5f, 1.0f};
    float bias = 0.0f;

    printf("--- TinyML Tensor Sistemi (Negatif Deger Testi) ---\n\n");

    // Ağırlıkları Yükle
    TinyTensor weightTensor;
    quantize_tensor(&weightTensor, trained_weights, n);

    // Kullanıcıdan Veri Al (Negatif değer girmesini iste)
    float user_input[3];
    printf("Lutfen en az bir tane NEGATIF sayi girin (Orn: -2.5, 3.0, -0.4):\n");
    for(int i = 0; i < n; i++) {
        printf("Giris [%d]: ", i);
        while(scanf("%f", &user_input[i]) != 1) { printf("Hata! Sayi giriniz: "); clear_buffer(); }
    }

    // Veriyi Sıkıştır
    TinyTensor inputTensor;
    quantize_tensor(&inputTensor, user_input, n);

    // ADIM 1: HAM VERİYİ GÖSTER (Negatifler Burada Görünmeli!)
    printf("\n--- 1. HAM TENSOR VERISI (RAW DATA) ---\n");
    printf("Burada negatif sayilar bellekte int8 olarak saklanmaktadir:\n");
    printf("Index | Girilen (f32) | Bellek (i8) | Geri Okunan (f32)\n");
    printf("-------------------------------------------------------\n");
    for(int i = 0; i < n; i++) {
        printf("  %d   | %8.3f    |    %4d     | %8.3f\n",
               i, user_input[i], inputTensor.data.i8_ptr[i], get_tensor_value(&inputTensor, i));
    }

    // ADIM 2: RELU UYGULAMA (Şimdi Negatifler Sıfır Olacak)
    printf("\n--- 2. AKTIVASYON KATMANI (ReLU) ---\n");
    printf("Yapay zeka modeli negatifleri eliyor (Non-Linearity)...\n");

    apply_relu(&inputTensor);

    printf("Index | Onceki Hal    | ReLU Sonrasi (i8) | Sonuc (f32)\n");
    printf("-------------------------------------------------------\n");
    for(int i = 0; i < n; i++) {
         // Not: ReLU sonrası bellekteki değer değiştiği için tekrar okuyoruz
        float val = get_tensor_value(&inputTensor, i);
        printf("  %d   | %8.3f    |    %4d           | %8.3f\n",
               i, user_input[i], inputTensor.data.i8_ptr[i], val);
    }

    // Temizlik
    free(inputTensor.data.i8_ptr);
    free(weightTensor.data.i8_ptr);

    printf("\nProgram tamamlandi.\n");
    return 0;
}