/* EdgeVoice ESP32-WROOM-32 (38-Pin) Hardware & KWS System Configuration */
#ifndef CONFIG_H_
#define CONFIG_H_

#include <cstdint>

// --- Hardware Pinouts for ESP32-WROOM-32 (38-Pin DevKit) ---
#define I2S_PORT            I2S_NUM_0
#define I2S_BCLK_PIN        14  // Bit Clock (BCK)
#define I2S_WS_PIN          15  // Word Select / LRCK (WS)
#define I2S_DIN_PIN         32  // Data In / DOUT (SD)

// --- Audio Format ---
#define SAMPLE_RATE         16000
#define AUDIO_WINDOW_MS     1000
#define AUDIO_WINDOW_SAMPLES (SAMPLE_RATE * AUDIO_WINDOW_MS / 1000) // 16000 samples

// --- Mel Feature Extraction ---
#define FRAME_SIZE_MS       40
#define FRAME_STEP_MS       20
#define FRAME_SIZE_SAMPLES  (SAMPLE_RATE * FRAME_SIZE_MS / 1000)   // 640 samples
#define FRAME_STEP_SAMPLES  (SAMPLE_RATE * FRAME_STEP_MS / 1000)   // 320 samples
#define NUM_MEL_BINS        40
#define NUM_FRAMES          49
#define FEATURE_ELEMENTS    (NUM_FRAMES * NUM_MEL_BINS)            // 1960 values

// --- Model & TFLite Micro Arena ---
#define TENSOR_ARENA_SIZE   (40 * 1024)                            // 40 KB Arena
#define NUM_CLASSES         4

// --- Trigger Logic & Thresholds ---
#define CONFIDENCE_THRESHOLD 0.25   // DEBUG: lowered from 0.30 for easier triggering
#define CONSECUTIVE_REQUIRED 1
#define COOLDOWN_MS          800    // DEBUG: reduced from 2000ms for faster re-trigger

// --- System Budget Limits ---
#define TARGET_MAX_RAM_KB    256
#define TARGET_MAX_CPU_IDLE  10.0f

#endif // CONFIG_H_
