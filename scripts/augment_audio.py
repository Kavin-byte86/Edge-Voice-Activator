#!/usr/bin/env python3
"""
GroundWatch Audio Augmentation Library
Applies realistic time shifts, gain variations, noise injection, and reverberation.
"""

import numpy as np
import random
from scipy import signal

def add_background_noise(clean_audio, noise_audio, snr_db=10):
    """Mixes noise into clean audio at specified Signal-to-Noise Ratio (SNR)."""
    if len(noise_audio) < len(clean_audio):
        # Repeat noise if shorter
        tile_factor = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, tile_factor)
        
    start = random.randint(0, len(noise_audio) - len(clean_audio))
    noise_clip = noise_audio[start:start+len(clean_audio)]
    
    clean_power = np.mean(clean_audio ** 2) + 1e-12
    noise_power = np.mean(noise_clip ** 2) + 1e-12
    
    target_noise_power = clean_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    
    mixed = clean_audio + scale * noise_clip
    return np.clip(mixed, -1.0, 1.0)

def random_time_shift(samples, max_shift_samples=1600):
    """Shifts audio in time up to +/- max_shift_samples (e.g. 100ms at 16kHz)."""
    shift = random.randint(-max_shift_samples, max_shift_samples)
    if shift == 0:
        return samples
    elif shift > 0:
        return np.pad(samples[:-shift], (shift, 0), mode='constant')
    else:
        return np.pad(samples[-shift:], (0, -shift), mode='constant')

def apply_random_gain(samples, min_gain=0.6, max_gain=1.4):
    """Applies random volume scaling."""
    gain = random.uniform(min_gain, max_gain)
    return np.clip(samples * gain, -1.0, 1.0)

def apply_reverberation(samples, rt60=0.2, sr=16000):
    """Simple synthetic room impulse response (reverb) simulation."""
    len_ir = int(sr * rt60)
    decay = np.exp(-3.0 * np.linspace(0, 1, len_ir))
    ir = np.random.normal(0, 0.1, len_ir) * decay
    ir[0] = 1.0  # direct path
    ir = ir / np.linalg.norm(ir)
    
    reverberated = signal.convolve(samples, ir, mode='full')[:len(samples)]
    return np.clip(reverberated, -1.0, 1.0)

def augment_sample(samples, noise_samples_list=None, p_noise=0.4, p_shift=0.5, p_gain=0.5, p_reverb=0.2):
    """Applies a pipeline of random augmentations."""
    aug = samples.copy()
    
    if random.random() < p_shift:
        aug = random_time_shift(aug)
        
    if random.random() < p_gain:
        aug = apply_random_gain(aug)
        
    if noise_samples_list and random.random() < p_noise:
        noise = random.choice(noise_samples_list)
        snr = random.uniform(8.0, 20.0)
        aug = add_background_noise(aug, noise, snr_db=snr)
        
    if random.random() < p_reverb:
        aug = apply_reverberation(aug)
        
    return aug
