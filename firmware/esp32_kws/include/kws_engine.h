/* GroundWatch TFLite Micro KWS Engine */
#ifndef KWS_ENGINE_H_
#define KWS_ENGINE_H_

#include <cstdint>
#include <cstddef>

bool init_kws_engine();
bool run_kws_inference(const int8_t* int8_features, float* out_probabilities, int8_t* out_raw_output);

float get_input_scale();
int get_input_zero_point();

size_t get_tensor_arena_bytes();
size_t get_tensor_arena_used_bytes();

#endif // KWS_ENGINE_H_
