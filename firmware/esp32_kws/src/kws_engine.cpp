#include "kws_engine.h"
#include "config.h"
#include "model_data.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <esp_heap_caps.h>
#include <cmath>
#include <cstdlib>

static uint8_t* s_tensor_arena = nullptr;
static const tflite::Model* s_model = nullptr;
static tflite::MicroInterpreter* s_interpreter = nullptr;
static TfLiteTensor* s_input_tensor = nullptr;
static TfLiteTensor* s_output_tensor = nullptr;
static tflite::MicroErrorReporter s_error_reporter;
static size_t s_arena_used_bytes = 0;

bool init_kws_engine() {
    if (!s_tensor_arena) {
        // Safely allocate 16-byte aligned memory in internal DRAM for TFLite Micro tensor arena
        s_tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_tensor_arena) {
            s_tensor_arena = (uint8_t*)malloc(TENSOR_ARENA_SIZE);
            if (!s_tensor_arena) return false;
        }
    }

    s_model = tflite::GetModel(g_model);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        return false;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        s_model, resolver, s_tensor_arena, TENSOR_ARENA_SIZE, &s_error_reporter
    );
    s_interpreter = &static_interpreter;

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        return false;
    }

    s_arena_used_bytes = s_interpreter->arena_used_bytes();

    s_input_tensor = s_interpreter->input(0);
    s_output_tensor = s_interpreter->output(0);
    return true;
}

float get_input_scale() {
    return s_input_tensor ? s_input_tensor->params.scale : 1.0f;
}

int get_input_zero_point() {
    return s_input_tensor ? s_input_tensor->params.zero_point : 0;
}

bool run_kws_inference(const int8_t* int8_features, float* out_probabilities, int8_t* out_raw_output) {
    if (!s_interpreter || !s_input_tensor || !s_output_tensor) return false;

    int8_t* input_buffer = s_input_tensor->data.int8;
    for (int i = 0; i < FEATURE_ELEMENTS; i++) {
        input_buffer[i] = int8_features[i];
    }

    if (s_interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    int8_t* output_buffer = s_output_tensor->data.int8;
    float out_scale = s_output_tensor->params.scale;
    int out_zero_point = s_output_tensor->params.zero_point;

    float exp_sum = 0.0f;
    float logits[NUM_CLASSES];

    for (int i = 0; i < NUM_CLASSES; i++) {
        if (out_raw_output) out_raw_output[i] = output_buffer[i];
        float dequantized = (output_buffer[i] - out_zero_point) * out_scale;
        logits[i] = dequantized;
        exp_sum += expf(dequantized);
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        out_probabilities[i] = expf(logits[i]) / exp_sum;
    }

    return true;
}

size_t get_tensor_arena_bytes() {
    return TENSOR_ARENA_SIZE;
}

size_t get_tensor_arena_used_bytes() {
    return s_arena_used_bytes;
}
