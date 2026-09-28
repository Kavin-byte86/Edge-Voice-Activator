#!/usr/bin/env python3
"""
GroundWatch Dataset Validator
Validates format, channels, sample rate, duration, and integrity of audio files.
"""

import os
import glob
import yaml
import scipy.io.wavfile as wav
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

EXPECTED_SR = config["audio"]["sample_rate"]
EXPECTED_SAMPLES = int(EXPECTED_SR * config["audio"]["window_duration_sec"])

DATASET_RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "raw"))

def validate_audio_file(filepath):
    try:
        sr, audio = wav.read(filepath)
        if sr != EXPECTED_SR:
            return False, f"Sample rate mismatch: {sr} != {EXPECTED_SR}"
        
        if audio.ndim > 1 and audio.shape[1] > 1:
            return False, f"Channels > 1: {audio.shape}"
            
        if len(audio) != EXPECTED_SAMPLES:
            return False, f"Length mismatch: {len(audio)} != {EXPECTED_SAMPLES}"
            
        if np.isnan(audio).any() or np.isinf(audio).any():
            return False, "NaN or Inf detected in audio samples"
            
        return True, "Valid"
    except Exception as e:
        return False, f"Read error: {str(e)}"

def main():
    print("=== GroundWatch Dataset Validation Starting ===")
    all_files = glob.glob(os.path.join(DATASET_RAW, "**", "*.wav"), recursive=True)
    print(f"Found {len(all_files)} total WAV files to validate.")
    
    valid_count = 0
    invalid_count = 0
    errors = []
    
    for fpath in all_files:
        is_valid, msg = validate_audio_file(fpath)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors.append((fpath, msg))
            
    print(f"Validation Complete!")
    print(f"  Valid files:   {valid_count}")
    print(f"  Invalid files: {invalid_count}")
    
    if invalid_count > 0:
        print("\nErrors encountered:")
        for fp, err in errors[:10]:
            print(f"  {os.path.basename(fp)}: {err}")
        raise ValueError(f"Dataset validation failed for {invalid_count} files!")
    else:
        print("SUCCESS: All dataset samples meet the 16kHz Mono 1.0s WAV requirement.")

if __name__ == "__main__":
    main()
