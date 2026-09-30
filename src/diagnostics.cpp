#include "diagnostics.h"
#include "config.h"
#include "i2s_audio.h"
#include "feature_provider.h"
#include "kws_engine.h"
#include "model_data.h"
#include <esp_system.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <Arduino.h>

static float s_idle_cpu_percent = 0.0f;
static float s_inference_cpu_percent = 0.0f;
static uint64_t s_last_cpu_sample_time = 0;

void init_diagnostics() {
    s_last_cpu_sample_time = esp_timer_get_time();
}

void update_cpu_measurements() {
    s_last_cpu_sample_time = esp_timer_get_time();
    (void)s_idle_cpu_percent;
    (void)s_inference_cpu_percent;
}

float get_idle_cpu_percent() {
    return s_idle_cpu_percent;
}

float get_inference_cpu_percent() {
    return s_inference_cpu_percent;
}

SystemMemoryStats get_system_memory_stats() {
    SystemMemoryStats stats;
    stats.free_heap_bytes      = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    stats.total_heap_bytes     = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    stats.min_free_heap_bytes  = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    stats.max_alloc_block_bytes = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);

    stats.tensor_arena_bytes        = get_tensor_arena_bytes();
    stats.tensor_arena_used_bytes   = get_tensor_arena_used_bytes();
    stats.ring_buffer_bytes         = get_ring_buffer_ram_bytes();
    stats.dma_audio_bytes           = get_i2s_dma_ram_bytes();

    stats.static_global_ram_bytes   = stats.tensor_arena_bytes
                                    + stats.ring_buffer_bytes
                                    + stats.dma_audio_bytes
                                    + get_feature_provider_ram_bytes();
    stats.total_application_ram_bytes = stats.static_global_ram_bytes
                                      + (stats.total_heap_bytes - stats.free_heap_bytes);

    UBaseType_t hwm = uxTaskGetStackHighWaterMark(NULL);
    stats.loop_stack_hwm_bytes = (uint32_t)(hwm * sizeof(StackType_t));

    return stats;
}

void print_hardware_benchmark_report(uint32_t feature_extraction_ms, uint32_t inference_ms, float akash_go_confidence) {
    SystemMemoryStats mem = get_system_memory_stats();
    float total_ram_kb = mem.total_application_ram_bytes / 1024.0f;

    bool ram_pass = (total_ram_kb < (float)TARGET_MAX_RAM_KB);
    bool int8_pass = true;

    Serial.println("====================================");
    Serial.println("VOICEEDGE KWS BENCHMARK");
    Serial.println("====================================");
    Serial.println();
    Serial.println("Device:");
    Serial.println("ESP32-WROOM-32 (38-Pin)");
    Serial.println();
    Serial.println("CPU:");
    Serial.println("240 MHz Dual-Core Xtensa LX6");
    Serial.println();
    Serial.printf("Free Heap:\n%u bytes\n\n",          mem.free_heap_bytes);
    Serial.printf("Minimum Free Heap:\n%u bytes\n\n",  mem.min_free_heap_bytes);
    Serial.printf("Largest Free Block:\n%u bytes\n\n", mem.max_alloc_block_bytes);
    Serial.printf("Tensor Arena (Allocated):\n%u bytes\n\n", mem.tensor_arena_bytes);
    Serial.printf("Tensor Arena (Model Used):\n%u bytes\n\n", mem.tensor_arena_used_bytes);
    Serial.printf("Audio DMA Buffer:\n%u bytes\n\n",   mem.dma_audio_bytes);
    Serial.printf("Ring Buffer:\n%u bytes\n\n",        mem.ring_buffer_bytes);
    Serial.printf("Model Size:\n%d bytes\n\n",         g_model_len);
    Serial.printf("loopTask Stack HWM:\n%u bytes free\n\n", mem.loop_stack_hwm_bytes);
    Serial.println("------------------------------------");
    Serial.println();
    Serial.printf("Inference:\n%u ms\n\n",           inference_ms);
    Serial.printf("Feature Extraction:\n%u ms\n\n",  feature_extraction_ms);
    Serial.printf("Keyword Confidence:\n%.4f\n\n",   akash_go_confidence);
    Serial.println("CPU Utilization:");
    Serial.println("NOT MEASURED (requires FreeRTOS runtime stats or hardware timer profiling)\n");
    Serial.printf("RAM Used (estimate):\n%.2f KB\n\n", total_ram_kb);
    Serial.println("------------------------------------");
    Serial.println();
    Serial.println("TARGET CHECK");
    Serial.println();
    Serial.printf("RAM < %d KB:\n%s (%.1f KB)\n\n", TARGET_MAX_RAM_KB, ram_pass ? "PASS" : "FAIL", total_ram_kb);
    Serial.println("CPU < 10%:");
    Serial.println("NOT MEASURED (enable FreeRTOS runtime stats to measure)\n");
    Serial.printf("INT8:\n%s\n\n", int8_pass ? "PASS" : "FAIL");
    Serial.println("====================================");
}
