#include "i2s_audio.h"
#include "config.h"
#include <driver/i2s.h>
#include <cstring>
#include <cstdlib>
#include <Arduino.h>

static int16_t* s_ring_buffer = nullptr;
static size_t s_ring_head = 0;

bool init_i2s_audio() {
    if (!s_ring_buffer) {
        s_ring_buffer = (int16_t*)malloc(AUDIO_WINDOW_SAMPLES * sizeof(int16_t));
        if (!s_ring_buffer) return false;
    }
    std::memset(s_ring_buffer, 0, AUDIO_WINDOW_SAMPLES * sizeof(int16_t));
    s_ring_head = 0;

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DIN_PIN
    };

    if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) != ESP_OK) {
        return false;
    }
    if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
        return false;
    }
    return true;
}

int read_i2s_samples(int16_t* dest_buffer, size_t max_samples) {
    static int32_t raw_i2s_buf[512];
    size_t bytes_read = 0;
    
    size_t bytes_to_read = max_samples * sizeof(int32_t);
    if (bytes_to_read > sizeof(raw_i2s_buf)) {
        bytes_to_read = sizeof(raw_i2s_buf);
    }

    esp_err_t res = i2s_read(I2S_PORT, raw_i2s_buf, bytes_to_read, &bytes_read, pdMS_TO_TICKS(100));
    if (res != ESP_OK || bytes_read == 0) {
        return 0;
    }

    size_t samples_read = bytes_read / sizeof(int32_t);
    for (size_t i = 0; i < samples_read; i++) {
        dest_buffer[i] = (int16_t)(raw_i2s_buf[i] >> 14);
    }
    return samples_read;
}

void update_ring_buffer(const int16_t* new_samples, size_t count) {
    if (!s_ring_buffer) return;
    for (size_t i = 0; i < count; i++) {
        s_ring_buffer[s_ring_head] = new_samples[i];
        s_ring_head = (s_ring_head + 1) % AUDIO_WINDOW_SAMPLES;
    }
}

void get_latest_window(int16_t* out_16000_samples) {
    if (!s_ring_buffer) return;
    size_t first_part = AUDIO_WINDOW_SAMPLES - s_ring_head;
    std::memcpy(out_16000_samples, &s_ring_buffer[s_ring_head], first_part * sizeof(int16_t));
    if (s_ring_head > 0) {
        std::memcpy(&out_16000_samples[first_part], &s_ring_buffer[0], s_ring_head * sizeof(int16_t));
    }
}

const int16_t* get_ring_buffer() {
    return s_ring_buffer;
}

size_t get_ring_buffer_head() {
    return s_ring_head;
}

size_t get_ring_buffer_ram_bytes() {
    return AUDIO_WINDOW_SAMPLES * sizeof(int16_t); // 32,000 bytes
}

size_t get_i2s_dma_ram_bytes() {
    return 4 * 512 * sizeof(int32_t); // 8 KB DMA buffer
}
