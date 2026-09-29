#include <Arduino.h>
#include "config.h"
#include "i2s_audio.h"
#include "feature_provider.h"
#include "kws_engine.h"
#include "diagnostics.h"

// s_audio_window is ELIMINATED: zero-copy ring buffer access saves 32,000 bytes
static int8_t s_int8_features[FEATURE_ELEMENTS];
static float s_class_probs[NUM_CLASSES];
static int8_t s_raw_outputs[NUM_CLASSES];

static int s_consecutive_detections = 0;
static uint32_t s_last_trigger_time = 0;
static uint32_t s_last_diag_time = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 2000);

    Serial.println("\n[EdgeVoice] ESP32-WROOM-32 Edge Voice Activator Starting...");
    Serial.printf("[EdgeVoice] Target Keyword: '%s'\n", "Akash Go");

    init_diagnostics();

    if (!init_i2s_audio()) {
        Serial.println("[ERROR] Failed to initialize INMP441 I2S Microphone!");
        while (1) delay(1000);
    }
    Serial.println("[OK] I2S INMP441 Microphone Initialized (16kHz 1-ch PCM).");

    if (!init_feature_provider()) {
        Serial.println("[ERROR] Failed to initialize DSP Feature Provider!");
        while (1) delay(1000);
    }
    Serial.println("[OK] Log-Mel DSP Feature Provider Initialized.");

    if (!init_kws_engine()) {
        Serial.println("[ERROR] Failed to initialize TFLite Micro KWS Engine!");
        while (1) delay(1000);
    }
    Serial.println("[OK] TFLite Micro INT8 Engine Initialized.");

    // Print initial memory snapshot so early heap problems and arena usage are visible.
    {
        SystemMemoryStats mem = get_system_memory_stats();
        Serial.println("\n[DIAG] Post-init memory snapshot:");
        Serial.printf("  Free Heap          : %u B\n", mem.free_heap_bytes);
        Serial.printf("  Min Free Heap      : %u B\n", mem.min_free_heap_bytes);
        Serial.printf("  Largest Free Block : %u B\n", mem.max_alloc_block_bytes);
        Serial.printf("  Tensor Arena Alloc : %u B\n", mem.tensor_arena_bytes);
        Serial.printf("  Tensor Arena Used  : %u B\n", mem.tensor_arena_used_bytes);
        Serial.printf("  Ring Buffer        : %u B\n", mem.ring_buffer_bytes);
        Serial.printf("  DMA Audio Buffer   : %u B\n", mem.dma_audio_bytes);
        Serial.printf("  loopTask Stack HWM : %u B free\n", mem.loop_stack_hwm_bytes);
    }

    Serial.println("\n[EdgeVoice] Continuous Listening Active...\n");
}

void loop() {
    // static: stays in .bss, not on the loopTask stack each call (saves 640 B).
    static int16_t s_chunk[FRAME_STEP_SAMPLES];
    static uint32_t s_loop_count = 0;
    s_loop_count++;

    if (s_loop_count <= 2) {
        Serial.printf("[DEBUG] loopTask #%u: reading I2S...\n", s_loop_count);
    }

    int samples_read = read_i2s_samples(s_chunk, FRAME_STEP_SAMPLES);

    if (s_loop_count <= 2) {
        Serial.printf("[DEBUG] loopTask #%u: I2S read %d samples\n", s_loop_count, samples_read);
    }

    static uint32_t s_recent_rms = 0;
    if (samples_read > 0) {
        update_ring_buffer(s_chunk, samples_read);

        // Compute RMS energy of incoming audio chunk to verify hardware mic signal
        int64_t sum_sq = 0;
        for (int i = 0; i < samples_read; i++) {
            int32_t val = s_chunk[i];
            sum_sq += val * val;
        }
        uint32_t chunk_rms = (uint32_t)sqrtf((float)(sum_sq / samples_read));
        s_recent_rms = (s_recent_rms * 7 + chunk_rms) / 8;
    }

    uint32_t now = millis();

    // Perform sliding window inference — run every 10ms for lower wake-word latency
    static uint32_t last_infer_time = 0;
    if (now - last_infer_time >= 10) {
        last_infer_time = now;

        // Zero-copy: read directly from the circular ring buffer without copying 32 KB!
        if (s_loop_count <= 2) {
            Serial.printf("[DEBUG] loopTask #%u: starting extract_features (zero-copy)...\n", s_loop_count);
        }
        uint32_t t_feat_start = micros();
        extract_features_from_ring_buffer(get_ring_buffer(), get_ring_buffer_head(),
                                         s_int8_features, get_input_scale(), get_input_zero_point());
        uint32_t t_feat_end = micros();

        if (s_loop_count <= 2) {
            Serial.printf("[DEBUG] loopTask #%u: extract_features done in %u ms\n",
                          s_loop_count, (t_feat_end - t_feat_start) / 1000);
            Serial.printf("[DEBUG] loopTask #%u: starting inference...\n", s_loop_count);
        }

        uint32_t t_infer_start = micros();
        run_kws_inference(s_int8_features, s_class_probs, s_raw_outputs);
        uint32_t t_infer_end = micros();

        uint32_t feat_ms  = (t_feat_end   - t_feat_start)  / 1000;
        uint32_t infer_ms = (t_infer_end  - t_infer_start) / 1000;

        if (s_loop_count <= 2) {
            Serial.printf("[DEBUG] loopTask #%u: inference done in %u ms (Akash Go prob: %.4f)\n",
                          s_loop_count, infer_ms, s_class_probs[0]);
        }

        float akash_go_prob = s_class_probs[0];

        // Trigger Smoothing Logic
        if (akash_go_prob >= CONFIDENCE_THRESHOLD) {
            s_consecutive_detections++;
        } else {
            s_consecutive_detections = 0;
        }

        // Keyword Trigger Event
        if (s_consecutive_detections >= CONSECUTIVE_REQUIRED) {
            if (now - s_last_trigger_time >= COOLDOWN_MS) {
                s_last_trigger_time = now;
                s_consecutive_detections = 0;

                Serial.println("\n>>> [KEYWORD DETECTED] 'Akash Go' <<<");
                Serial.printf("  Confidence: %.4f | Feat: %u ms | Infer: %u ms | RMS: %u\n",
                              akash_go_prob, feat_ms, infer_ms, s_recent_rms);

                // Output full benchmark report block
                print_hardware_benchmark_report(feat_ms, infer_ms, akash_go_prob);
            }
        }
    }

    // Periodic diagnostics: print on first loop and every 500ms during idle listening
    if (s_loop_count == 1 || (now - s_last_diag_time >= 500)) {
        s_last_diag_time = now;
        update_cpu_measurements();
        SystemMemoryStats mem = get_system_memory_stats();
        Serial.printf("[IDLE] FreeHeap: %u B | MinFreeHeap: %u B | LargestBlk: %u B"
                      " | ArenaAlloc: %u B | ArenaUsed: %u B | Stack HWM: %u B free"
                      " | RMS: %u%s\n",
                      mem.free_heap_bytes,
                      mem.min_free_heap_bytes,
                      mem.max_alloc_block_bytes,
                      mem.tensor_arena_bytes,
                      mem.tensor_arena_used_bytes,
                      mem.loop_stack_hwm_bytes,
                      s_recent_rms,
                      (s_recent_rms < 10) ? " (QUIET/CHECK_MIC)" : " (MIC_LIVE)");
    }

    vTaskDelay(1 / portTICK_PERIOD_MS);
}
