#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoiceEdge Dataset Generator - Real gTTS Speech Synthesis
=========================================================
Synthesises REAL human-like TTS audio using Google Text-to-Speech (gTTS).
All samples are 16kHz, 16-bit PCM Mono WAV, exactly 1.0 second.

IMPORTANT: The previous formant-based synthesizer was phrase-blind -- it produced
identical sawtooth buzz regardless of the text. This version uses actual TTS that
sounds like the target phrase.

Classes:
  positive  -> "Akash Go" and spelling variants, multiple accents
  negative  -> hard negatives phonetically similar to "Akash Go"
  unknown   -> generic unrelated phrases
  noise     -> synthetic white/pink/brown noise
  silence   -> near-zero noise floor
"""
import sys, io as _io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
import io
import time
import math
import random
import yaml
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import librosa

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SAMPLE_RATE = config["audio"]["sample_rate"]          # 16000
DURATION    = config["audio"]["window_duration_sec"]  # 1.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)          # 16000

DATASET_RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "raw"))
POS_DIR  = os.path.join(DATASET_RAW, "positive")
NEG_DIR  = os.path.join(DATASET_RAW, "negative")
UNK_DIR  = os.path.join(DATASET_RAW, "unknown")
NOISE_DIR= os.path.join(DATASET_RAW, "noise")

for d in [POS_DIR, NEG_DIR, UNK_DIR, NOISE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Phrase lists ──────────────────────────────────────────────────────────────

# Primary keyword and phonetic variants
POSITIVE_PHRASES = [
    "Akash Go",
    "Aakash Go",
    "Akash, go",
]

# Hard negatives: phonetically close but NOT the keyword
HARD_NEGATIVES = [
    "Akash",
    "Aakash",
    "Akash no",
    "Akash come",
    "Akash move",
    "Akash stop",
    "Akash run",
    "Akash now",
    "Akash go on",
    "go Akash",
    "just go",
    "a cash go",
    "ash go",
    "cash go",
]

# Completely unrelated words and phrases
UNKNOWN_WORDS = [
    "hey google", "alexa", "siri", "ok google",
    "light on", "light off", "turn off", "volume up", "volume down",
    "play music", "stop music", "next track", "pause",
    "good morning", "good night", "hello", "hi there",
    "one", "two", "three", "four", "five",
    "yes", "no", "stop", "go", "up", "down",
    "open door", "start engine", "power down",
    "machine learning", "neural network", "embedded system",
    "sensor data", "microphone test",
]

# gTTS accent TLDs — different regional accents from Google TTS
# Each TLD produces a perceptibly different voice/accent
TLDS = ["com", "co.in", "co.uk", "ca", "com.au", "ie", "co.za"]

# ── Audio helpers ─────────────────────────────────────────────────────────────

def gtts_to_float32(phrase: str, tld: str = "com", slow: bool = False,
                    retries: int = 4) -> np.ndarray | None:
    """
    Converts a text phrase to 16kHz float32 audio using gTTS + librosa.
    Returns None on repeated failure (network unavailable).
    Uses librosa for MP3 decoding — no ffmpeg required.
    """
    from gtts import gTTS
    for attempt in range(retries):
        try:
            buf = io.BytesIO()
            gTTS(text=phrase, lang="en", tld=tld, slow=slow).write_to_fp(buf)
            buf.seek(0)
            # librosa.load handles mp3 via audioread — no external tools needed
            y, _ = librosa.load(buf, sr=SAMPLE_RATE, mono=True)
            return y.astype(np.float32)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [gTTS retry {attempt+1}/{retries}] '{phrase}' tld={tld}: {e}. Waiting {wait}s…")
            time.sleep(wait)
    return None


def time_stretch_audio(samples: np.ndarray, rate: float) -> np.ndarray:
    """Stretch/compress audio speed without changing pitch (librosa PSOLA)."""
    if abs(rate - 1.0) < 0.01:
        return samples
    stretched = librosa.effects.time_stretch(y=samples, rate=rate)
    return stretched.astype(np.float32)


def fit_to_target_length(samples: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    """Centre-crop if too long, zero-pad with random offset if too short."""
    if len(samples) == target_len:
        return samples
    elif len(samples) > target_len:
        # Centre crop
        start = (len(samples) - target_len) // 2
        return samples[start:start + target_len]
    else:
        pad_total = target_len - len(samples)
        pad_left  = random.randint(0, pad_total)
        pad_right = pad_total - pad_left
        return np.pad(samples, (pad_left, pad_right), mode='constant')


def apply_random_gain(samples: np.ndarray,
                      min_gain: float = 0.70,
                      max_gain: float = 1.15) -> np.ndarray:
    gain = random.uniform(min_gain, max_gain)
    return np.clip(samples * gain, -1.0, 1.0)


def save_wav(filepath: str, float_samples: np.ndarray) -> None:
    int16 = np.clip(float_samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wav.write(filepath, SAMPLE_RATE, int16)


def generate_synthetic_noise(noise_type: str = "white",
                              num_samples: int = TARGET_SAMPLES) -> np.ndarray:
    if noise_type == "white":
        n = np.random.normal(0, 0.12, num_samples)
    elif noise_type == "pink":
        raw = np.random.normal(0, 0.12, num_samples)
        b, a = signal.butter(1, 0.1, btype='low')
        n = signal.lfilter(b, a, raw)
    elif noise_type == "brown":
        raw = np.random.normal(0, 0.08, num_samples)
        n = np.cumsum(raw) * 0.01
    else:
        n = np.zeros(num_samples)

    max_val = np.max(np.abs(n))
    if max_val > 0:
        n = n / max_val * random.uniform(0.02, 0.20)
    return n.astype(np.float32)


# ── Speed rates for augmentation ──────────────────────────────────────────────
SPEED_RATES = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd

    print("=" * 60)
    print("VoiceEdge Real-Speech Dataset Synthesis (gTTS + librosa)")
    print("=" * 60)

    metadata      = []
    speaker_map   = {}
    speaker_counter = 0
    failed_phrases = []

    def get_speaker_id(tld: str, slow: bool, rate: float) -> str:
        nonlocal speaker_counter
        key = f"gtts_{tld}_slow{int(slow)}_r{rate:.2f}"
        if key not in speaker_map:
            speaker_counter += 1
            speaker_map[key] = f"spk_{speaker_counter:04d}"
        return speaker_map[key]

    # ── POSITIVE samples ──────────────────────────────────────────────────────
    print("\n[1/4] Synthesising POSITIVE samples ('Akash Go')...")
    pos_idx = 0
    target_pos = config["dataset"]["positive_target"]  # 600 by default

    # We need (phrase × tld × slow × speed_rate) combinations
    # Enumerate deterministically then shuffle
    pos_combos = []
    for phrase in POSITIVE_PHRASES:
        for tld in TLDS:
            for slow in [False, True]:
                for rate in SPEED_RATES:
                    pos_combos.append((phrase, tld, slow, rate))

    random.shuffle(pos_combos)

    # Cache gTTS fetches to avoid re-fetching the same (phrase,tld,slow) triple
    gtts_cache: dict[tuple, np.ndarray | None] = {}

    for phrase, tld, slow, rate in pos_combos:
        if pos_idx >= target_pos:
            break

        fname  = f"pos_akash_go_{pos_idx:04d}.wav"
        fpath  = os.path.join(POS_DIR, fname)

        # Skip already-generated files (resume support)
        if os.path.exists(fpath):
            spk_id = get_speaker_id(tld, slow, rate)
            metadata.append({
                "filename":   fname, "filepath":   fpath,
                "label":      "akash_go", "speaker_id": spk_id,
                "phrase":     phrase, "category":   "positive",
            })
            pos_idx += 1
            continue

        cache_key = (phrase, tld, slow)
        if cache_key not in gtts_cache:
            print("  Fetching TTS: '%s' tld=%s slow=%s" % (phrase, tld, slow))
            gtts_cache[cache_key] = gtts_to_float32(phrase, tld=tld, slow=slow)

        base_audio = gtts_cache[cache_key]
        if base_audio is None:
            failed_phrases.append(cache_key)
            continue

        # Time-stretch then fit to 1 second
        stretched   = time_stretch_audio(base_audio, rate)
        final_audio = fit_to_target_length(stretched)
        final_audio = apply_random_gain(final_audio)

        spk_id = get_speaker_id(tld, slow, rate)
        save_wav(fpath, final_audio)

        metadata.append({
            "filename":   fname, "filepath":   fpath,
            "label":      "akash_go", "speaker_id": spk_id,
            "phrase":     phrase, "category":   "positive",
        })
        pos_idx += 1

    print("  -> Positive samples created: %d" % pos_idx)

    # ── HARD NEGATIVE samples ─────────────────────────────────────────────────
    print("\n[2/4] Synthesising HARD NEGATIVE samples...")
    neg_idx = 0

    neg_cache = {}

    for phrase in HARD_NEGATIVES:
        for tld in TLDS[:4]:          # 4 accents per hard-negative phrase
            for slow in [False]:
                for rate in [0.90, 1.0, 1.10]:
                    cache_key = (phrase, tld, slow)
                    if cache_key not in neg_cache:
                        print("  Fetching TTS: '%s' tld=%s" % (phrase, tld))
                        neg_cache[cache_key] = gtts_to_float32(phrase, tld=tld, slow=slow)

                    base_audio = neg_cache[cache_key]
                    if base_audio is None:
                        continue

                    fname = "neg_hardneg_%04d.wav" % neg_idx
                    fpath = os.path.join(NEG_DIR, fname)

                    if not os.path.exists(fpath):
                        stretched   = time_stretch_audio(base_audio, rate)
                        final_audio = fit_to_target_length(stretched)
                        final_audio = apply_random_gain(final_audio)
                        save_wav(fpath, final_audio)
                    else:
                        final_audio = np.zeros(TARGET_SAMPLES, dtype=np.float32)  # placeholder for metadata

                    spk_id = get_speaker_id(tld, slow, rate)
                    metadata.append({
                        "filename": fname, "filepath": fpath,
                        "label": "unknown", "speaker_id": spk_id,
                        "phrase": phrase, "category": "hard_negative",
                    })
                    neg_idx += 1

    print("  -> Hard negative samples created: %d" % neg_idx)

    # ── UNKNOWN samples ───────────────────────────────────────────────────────
    print("\n[3/4] Synthesising UNKNOWN word samples...")
    unk_file_idx = 0
    target_unk = config["dataset"]["unknown_target"]  # 1500

    unk_cache = {}
    unk_combos = []
    for phrase in UNKNOWN_WORDS:
        for tld in TLDS[:3]:          # 3 accents
            for rate in [0.90, 1.0, 1.10]:
                unk_combos.append((phrase, tld, False, rate))

    random.shuffle(unk_combos)

    for phrase, tld, slow, rate in unk_combos:
        if unk_file_idx >= target_unk:
            break

        fname = "unk_general_%04d.wav" % unk_file_idx
        fpath = os.path.join(UNK_DIR, fname)

        cache_key = (phrase, tld, slow)
        if cache_key not in unk_cache:
            print("  Fetching TTS: '%s' tld=%s" % (phrase, tld))
            unk_cache[cache_key] = gtts_to_float32(phrase, tld=tld, slow=slow)

        base_audio = unk_cache[cache_key]
        if base_audio is None:
            continue

        if not os.path.exists(fpath):
            stretched   = time_stretch_audio(base_audio, rate)
            final_audio = fit_to_target_length(stretched)
            final_audio = apply_random_gain(final_audio)
            save_wav(fpath, final_audio)

        spk_id = get_speaker_id(tld, slow, rate)
        metadata.append({
            "filename": fname, "filepath": fpath,
            "label": "unknown", "speaker_id": spk_id,
            "phrase": phrase, "category": "unknown_word",
        })
        unk_file_idx += 1

    print("  -> Unknown samples created: %d" % unk_file_idx)

    # ── NOISE & SILENCE samples ───────────────────────────────────────────────
    print("\n[4/4] Synthesising NOISE and SILENCE samples...")
    noise_target   = config["dataset"]["noise_target"]    # 500
    silence_target = config["dataset"]["silence_target"]  # 300

    for i in range(noise_target):
        ntype = ["white", "pink", "brown"][i % 3]
        fname = "noise_%04d.wav" % i
        fpath = os.path.join(NOISE_DIR, fname)
        if not os.path.exists(fpath):
            n_audio = generate_synthetic_noise(ntype)
            save_wav(fpath, n_audio)
        metadata.append({
            "filename": fname, "filepath": fpath,
            "label": "noise", "speaker_id": "env_noise",
            "phrase": "noise_%s" % ntype, "category": "noise",
        })

    for i in range(silence_target):
        fname = "silence_%04d.wav" % i
        fpath = os.path.join(NOISE_DIR, fname)
        if not os.path.exists(fpath):
            s_audio = np.random.normal(0, 0.001, TARGET_SAMPLES).astype(np.float32)
            save_wav(fpath, s_audio)
        metadata.append({
            "filename": fname, "filepath": fpath,
            "label": "silence", "speaker_id": "env_silence",
            "phrase": "silence", "category": "silence",
        })

    print("  -> Noise samples: %d, Silence samples: %d" % (noise_target, silence_target))

    # ── Save metadata ─────────────────────────────────────────────────────────
    df = pd.DataFrame(metadata)
    meta_path = os.path.join(DATASET_RAW, "metadata.csv")
    df.to_csv(meta_path, index=False)

    print("\n" + "=" * 60)
    print("Dataset Synthesis Completed!")
    print("  Total samples:          %d" % len(df))
    print("  Unique TTS speaker IDs: %d" % len(speaker_map))
    print("  Failed TTS fetches:     %d" % len(failed_phrases))
    print("  Metadata: %s" % meta_path)
    print()
    print(df["label"].value_counts())
    if failed_phrases:
        print("\n  WARNING: %d phrase(s) failed to fetch. Check network and re-run." % len(failed_phrases))


if __name__ == "__main__":
    main()
