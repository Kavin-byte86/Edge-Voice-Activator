/* GroundWatch Embedded DSP Feature Provider */
#ifndef FEATURE_PROVIDER_H_
#define FEATURE_PROVIDER_H_

#include <cstdint>
#include <cstddef>

bool init_feature_provider();

// Zero-copy: extracts features directly from the circular ring buffer without copying 32 KB
bool extract_features_from_ring_buffer(const int16_t* ring_buffer, size_t ring_head,
                                      int8_t* out_int8_features, float input_scale, int input_zero_point);

// Contiguous buffer version (for standalone audio arrays or tests)
bool extract_features_from_audio(const int16_t* audio_16000_samples,
                                int8_t* out_int8_features, float input_scale, int input_zero_point);

size_t get_feature_provider_ram_bytes();

#endif // FEATURE_PROVIDER_H_
