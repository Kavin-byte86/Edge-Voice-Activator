#include "feature_provider.h"
#include "config.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

static float s_hanning_window[FRAME_SIZE_SAMPLES]; // 2,560 B in .bss
static float s_mel_fbank[NUM_MEL_BINS][257];        // 41,120 B in .bss
static bool s_feature_provider_initialized = false;

// ---------------------------------------------------------------------------
// Large working buffers kept as file-scope statics so they live in .bss and
// never consume loopTask stack space.  Each call to extract_features_from_audio()
// reuses the same memory; the function is only ever called from one task.
//   s_frame_windowed  : 640  floats = 2,560 bytes
//   s_power_spectrum  : 257  floats = 1,028 bytes
//   s_raw_spectrogram : 49*40 floats = 7,840 bytes
// ---------------------------------------------------------------------------
static float s_frame_windowed[FRAME_SIZE_SAMPLES];          // 2,560 B in .bss
static float s_power_spectrum[257];                          // 1,028 B in .bss
static float s_raw_spectrogram[NUM_FRAMES][NUM_MEL_BINS];   // 7,840 B in .bss

static void hz_to_mel(float hz, float* mel) {
    *mel = 2595.0f * log10f(1.0f + hz / 700.0f);
}

static void mel_to_hz(float mel, float* hz) {
    *hz = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

extern "C" {
    struct kiss_fftr_state;
    typedef struct kiss_fftr_state* kiss_fftr_cfg;
    typedef float kiss_fft_scalar;
    typedef struct {
        kiss_fft_scalar r;
        kiss_fft_scalar i;
    } kiss_fft_cpx;

    kiss_fftr_cfg kiss_fftr_alloc(int nfft, int inverse_fft, void* mem, size_t* lenmem);
    void kiss_fftr(kiss_fftr_cfg cfg, const kiss_fft_scalar* timedata, kiss_fft_cpx* freqdata);
    void kiss_fftr_free(void* p);
}

static kiss_fftr_cfg s_fft_cfg = nullptr;
static kiss_fft_cpx s_fft_out[257]; // 257 complex frequency bins
static void* s_fft_mem = nullptr;

static void fast_fft_257(const float* windowed_input, float* power_spectrum) {
    kiss_fftr(s_fft_cfg, (const kiss_fft_scalar*)windowed_input, s_fft_out);
    for (int k = 0; k < 257; k++) {
        power_spectrum[k] = (s_fft_out[k].r * s_fft_out[k].r + s_fft_out[k].i * s_fft_out[k].i) / 512.0f;
    }
}

bool init_feature_provider() {
    if (s_feature_provider_initialized) return true;

    // 1. Initialize Hanning Window
    for (int i = 0; i < FRAME_SIZE_SAMPLES; i++) {
        s_hanning_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FRAME_SIZE_SAMPLES - 1)));
    }

    // 2. Initialize Mel Filterbank Matrix (20 Hz to 7500 Hz)
    float min_mel, max_mel;
    hz_to_mel(20.0f, &min_mel);
    hz_to_mel(7500.0f, &max_mel);

    float mel_points[NUM_MEL_BINS + 2];
    float hz_points[NUM_MEL_BINS + 2];
    int bin_points[NUM_MEL_BINS + 2];

    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        mel_points[i] = min_mel + i * (max_mel - min_mel) / (NUM_MEL_BINS + 1);
        mel_to_hz(mel_points[i], &hz_points[i]);
        bin_points[i] = (int)floorf((512 + 1) * hz_points[i] / SAMPLE_RATE);
    }

    std::memset(s_mel_fbank, 0, NUM_MEL_BINS * 257 * sizeof(float));

    for (int m = 1; m <= NUM_MEL_BINS; m++) {
        int f_m_minus = bin_points[m - 1];
        int f_m = bin_points[m];
        int f_m_plus = bin_points[m + 1];

        for (int k = f_m_minus; k < f_m; k++) {
            if (k < 257) {
                s_mel_fbank[m - 1][k] = (float)(k - f_m_minus) / (float)(f_m - f_m_minus);
            }
        }
        for (int k = f_m; k < f_m_plus; k++) {
            if (k < 257) {
                s_mel_fbank[m - 1][k] = (float)(f_m_plus - k) / (float)(f_m_plus - f_m);
            }
        }
    }

    // 3. Initialize KissFFT Real 512-point FFT configuration
    if (!s_fft_cfg) {
        size_t memneeded = 0;
        kiss_fftr_alloc(512, 0, NULL, &memneeded);
        if (!s_fft_mem) {
            s_fft_mem = malloc(memneeded);
            if (!s_fft_mem) return false;
        }
        s_fft_cfg = kiss_fftr_alloc(512, 0, s_fft_mem, &memneeded);
        if (!s_fft_cfg) return false;
    }

    s_feature_provider_initialized = true;
    return true;
}

bool extract_features_from_ring_buffer(const int16_t* ring_buffer, size_t ring_head,
                                      int8_t* out_int8_features, float input_scale, int input_zero_point) {
    if (!s_feature_provider_initialized || !s_fft_cfg || !ring_buffer) return false;

    // Use file-scope static buffers (zero stack cost) instead of VLAs/locals.
    float* frame_windowed      = s_frame_windowed;
    float* power_spectrum      = s_power_spectrum;
    // s_raw_spectrogram is accessed via its own name below.

    float sum_val = 0.0f;
    int total_vals = NUM_FRAMES * NUM_MEL_BINS;

    for (int frame_idx = 0; frame_idx < NUM_FRAMES; frame_idx++) {
        int start_sample = frame_idx * FRAME_STEP_SAMPLES;
        
        for (int n = 0; n < FRAME_SIZE_SAMPLES; n++) {
            size_t sample_idx = (ring_head + start_sample + n) % AUDIO_WINDOW_SAMPLES;
            float norm_sample = (float)ring_buffer[sample_idx] / 32768.0f;
            frame_windowed[n] = norm_sample * s_hanning_window[n];
        }

        fast_fft_257(frame_windowed, power_spectrum);

        for (int mel_idx = 0; mel_idx < NUM_MEL_BINS; mel_idx++) {
            float mel_energy = 0.0f;
            for (int k = 0; k < 257; k++) {
                mel_energy += s_mel_fbank[mel_idx][k] * power_spectrum[k];
            }
            float log_mel = logf(mel_energy + 1e-6f);
            s_raw_spectrogram[frame_idx][mel_idx] = log_mel;
            sum_val += log_mel;
        }
    }

    float mean_val = sum_val / total_vals;
    float sq_diff_sum = 0.0f;
    for (int f = 0; f < NUM_FRAMES; f++) {
        for (int m = 0; m < NUM_MEL_BINS; m++) {
            float diff = s_raw_spectrogram[f][m] - mean_val;
            sq_diff_sum += diff * diff;
        }
    }
    float std_val = sqrtf(sq_diff_sum / total_vals);
    if (std_val < 1e-5f) std_val = 1.0f;

    // Apply z-score normalization & INT8 quantization
    for (int f = 0; f < NUM_FRAMES; f++) {
        for (int m = 0; m < NUM_MEL_BINS; m++) {
            float norm_val = (s_raw_spectrogram[f][m] - mean_val) / std_val;
            int quantized_val = (int)roundf(norm_val / input_scale) + input_zero_point;
            if (quantized_val < -128) quantized_val = -128;
            if (quantized_val > 127)  quantized_val = 127;

            out_int8_features[f * NUM_MEL_BINS + m] = (int8_t)quantized_val;
        }
    }
    return true;
}

bool extract_features_from_audio(const int16_t* audio_16000_samples, int8_t* out_int8_features, float input_scale, int input_zero_point) {
    // Contiguous buffer has ring_head = 0
    return extract_features_from_ring_buffer(audio_16000_samples, 0, out_int8_features, input_scale, input_zero_point);
}

size_t get_feature_provider_ram_bytes() {
    // Hanning window heap alloc  :  640 * 4 =  2,560 B
    // Mel filterbank heap alloc  : 40*257*4 = 41,120 B
    // Static working buffers (.bss):
    //   s_frame_windowed          :  640 * 4 =  2,560 B
    //   s_power_spectrum          :  257 * 4 =  1,028 B
    //   s_raw_spectrogram         : 49*40*4  =  7,840 B
    // Total reported (heap + bss working buffers):
    return (FRAME_SIZE_SAMPLES * sizeof(float))          // hanning (heap)
         + (NUM_MEL_BINS * 257 * sizeof(float))          // mel fbank (heap)
         + sizeof(s_frame_windowed)                       // working buf (.bss)
         + sizeof(s_power_spectrum)                       // working buf (.bss)
         + sizeof(s_raw_spectrogram);                     // working buf (.bss)
}
