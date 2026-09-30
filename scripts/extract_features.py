#!/usr/bin/env python3
"""
VoiceEdge Feature Extraction Library
Computes 49x40 Normalized Log-Mel Spectrogram features from 16kHz WAV audio.
Designed to match embedded ESP32 feature provider DSP output.
"""

import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft

def get_mel_filterbank(num_mel_bins=40, fft_len=512, sample_rate=16000, lower_freq=20.0, upper_freq=7500.0):
    """Generates triangular Mel filterbank matrix matching standard librosa/kaldi formulation."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    lower_mel = hz_to_mel(lower_freq)
    upper_mel = hz_to_mel(upper_freq)
    mel_points = np.linspace(lower_mel, upper_mel, num_mel_bins + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((fft_len + 1) * hz_points / sample_rate).astype(int)

    num_fft_bins = fft_len // 2 + 1
    fbank = np.zeros((num_mel_bins, num_fft_bins), dtype=np.float32)

    for m in range(1, num_mel_bins + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if k < num_fft_bins:
                fbank[m - 1, k] = (k - f_m_minus) / max(1, (f_m - f_m_minus))
        for k in range(f_m, f_m_plus):
            if k < num_fft_bins:
                fbank[m - 1, k] = (f_m_plus - k) / max(1, (f_m_plus - f_m))

    return fbank

FBANK = get_mel_filterbank(num_mel_bins=40, fft_len=512, sample_rate=16000, lower_freq=20.0, upper_freq=7500.0)
WINDOW = np.hanning(640).astype(np.float32)

def compute_log_mel_spectrogram(samples, sample_rate=16000, frame_size=640, frame_step=320, num_mel_bins=40):
    """
    Computes Normalized Log-Mel Spectrogram matrix.
    Input: 16000 samples (1D float array [-1, 1])
    Output: 2D numpy array of shape (49, 40)
    """
    if len(samples) < frame_size:
        samples = np.pad(samples, (0, frame_size - len(samples)), mode='constant')

    num_frames = (len(samples) - frame_size) // frame_step + 1
    frames = []

    num_fft_bins = 512 // 2 + 1

    for i in range(num_frames):
        start = i * frame_step
        frame_samples = samples[start:start+frame_size]
        
        windowed = frame_samples * WINDOW
        
        fft_input = windowed[:512]
        spectrum = np.abs(fft(fft_input)[:num_fft_bins])
        power_spectrum = (spectrum ** 2) / 512.0
        
        mel_energies = np.dot(FBANK, power_spectrum)
        log_mel = np.log(mel_energies + 1e-6)
        frames.append(log_mel)

    spectrogram = np.array(frames, dtype=np.float32)
    
    # Feature standardization (z-score normalization per sample)
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    if std < 1e-5:
        std = 1.0
    norm_spectrogram = (spectrogram - mean) / std
    return norm_spectrogram  # Shape: (49, 40)

def extract_file_features(filepath):
    """Loads 16kHz WAV and extracts (49, 40, 1) feature matrix."""
    sr, audio = wav.read(filepath)
    float_samples = audio.astype(np.float32) / 32768.0
    
    if len(float_samples) != 16000:
        if len(float_samples) > 16000:
            float_samples = float_samples[:16000]
        else:
            float_samples = np.pad(float_samples, (0, 16000 - len(float_samples)), mode='constant')
            
    spec = compute_log_mel_spectrogram(float_samples)
    return np.expand_dims(spec, axis=-1)  # Shape: (49, 40, 1)

if __name__ == "__main__":
    dummy_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
    spec = compute_log_mel_spectrogram(dummy_audio)
    print(f"Computed Standardized Spectrogram Shape: {spec.shape}")
    print(f"Feature Mean: {spec.mean():.4f}, Std: {spec.std():.4f}")
