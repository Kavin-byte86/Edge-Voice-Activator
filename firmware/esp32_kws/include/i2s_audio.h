/* GroundWatch I2S Audio Reader & Ring Buffer */
#ifndef I2S_AUDIO_H_
#define I2S_AUDIO_H_

#include <cstdint>
#include <cstddef>

bool init_i2s_audio();
int read_i2s_samples(int16_t* dest_buffer, size_t max_samples);
void update_ring_buffer(const int16_t* new_samples, size_t count);
void get_latest_window(int16_t* out_16000_samples);

// Zero-copy accessors to avoid copying the 32 KB audio window
const int16_t* get_ring_buffer();
size_t get_ring_buffer_head();

size_t get_ring_buffer_ram_bytes();
size_t get_i2s_dma_ram_bytes();

#endif // I2S_AUDIO_H_
