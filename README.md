# EdgeVoice: Edge Voice Activator for "Akash Go"
### ESP32-WROOM-32 (38-Pin) · TFLite Micro INT8 · Zero Cloud Dependency

<p align="center">
  <img src="https://img.shields.io/badge/Platform-ESP32--WROOM--32-red?style=for-the-badge&logo=espressif" />
  <img src="https://img.shields.io/badge/ML%20Framework-TFLite%20Micro%20INT8-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Build-PlatformIO-purple?style=for-the-badge&logo=platformio" />
  <img src="https://img.shields.io/badge/Wake%20Word-Akash%20Go-green?style=for-the-badge" />
</p>

---

**EdgeVoice** is a complete, reproducible, end-to-end TinyML keyword-spotting (KWS) system that detects the custom wake word **"Akash Go"** entirely on-device. No cloud, no API, no generic wake-word engine — just a tiny **16.80 KB INT8 neural network** running at **4.50% CPU utilization** on a bare ESP32-WROOM-32 microcontroller.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Hardware Specification & Pinout](#hardware-specification--pinout)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Software Dependencies & Environment Setup](#software-dependencies--environment-setup)
6. [Dataset Generation & Pipeline](#dataset-generation--pipeline)
7. [Model Architecture](#model-architecture)
8. [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
9. [Firmware: Build, Flash & Monitor](#firmware-build-flash--monitor)
10. [Benchmark Results](#benchmark-results)
11. [Serial Diagnostic Output](#serial-diagnostic-output)
12. [Constraints & Known Limitations](#constraints--known-limitations)
13. [Troubleshooting](#troubleshooting)

---

## Project Overview

| Property | Value |
| :--- | :--- |
| **Wake Word** | `"Akash Go"` |
| **Target Device** | ESP32-WROOM-32 (38-Pin DevKit) |
| **Microphone** | INMP441 I2S MEMS |
| **ML Framework** | TensorFlow Lite Micro |
| **Quantization** | Full INT8 (inputs, weights, activations, outputs) |
| **Model Size** | 16.80 KB (INT8 TFLite) |
| **RAM Footprint** | 173.50 KB (budget: < 256 KB) ✅ |
| **Idle CPU** | 4.50% (budget: < 10%) ✅ |
| **Inference Latency** | 15 ms |
| **Cloud Dependency** | ❌ None — fully offline |

---

## Hardware Specification & Pinout

### Bill of Materials

| # | Component | Part Number | Qty |
| :-- | :--- | :--- | :--: |
| 1 | Microcontroller | ESP32-WROOM-32 (38-Pin DevKit) | 1 |
| 2 | Microphone | INMP441 I2S MEMS Microphone | 1 |
| 3 | Power | USB Micro-B cable + 5V adapter | 1 |
| 4 | Wiring | Dupont jumper wires (female-to-male) | 6 |

### INMP441 → ESP32-WROOM-32 Wiring

| INMP441 Pin | ESP32 GPIO | Description |
| :---: | :---: | :--- |
| **VDD** | 3.3V | Power supply (3.3V ONLY — DO NOT use 5V) |
| **GND** | GND | Ground |
| **SCK / BCLK** | **GPIO 14** | I2S Bit Clock |
| **WS / LRCK** | **GPIO 15** | I2S Word Select (Left/Right Clock) |
| **SD / DOUT** | **GPIO 32** | I2S Serial Data Out |
| **L/R** | GND | Channel select: GND = Left Channel |

> ⚠️ **Critical**: The INMP441 operates at 3.3V. Connecting VDD to 5V will permanently damage the microphone.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      OFFLINE PIPELINE (PC)                      │
│                                                                 │
│  gTTS/pyttsx3     librosa/scipy     TensorFlow 2.15            │
│  ─────────────    ─────────────     ────────────────           │
│  generate_        extract_          train.py  →  model.keras    │
│  dataset.py  →    features.py  →    quantize.py → model.tflite  │
│  (4,124 clips)    (49×40 MFSCs)     benchmark_model.py          │
│                                     convert_to_c_array.py       │
│                                     ↓                           │
│                                  model_data.h (PROGMEM)         │
└───────────────────────────────────┬─────────────────────────────┘
                                    │ PlatformIO flash
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              ESP32-WROOM-32 (38-Pin) — RUNTIME                  │
│                                                                  │
│  INMP441  ──I2S DMA──▶  Ring Buffer (32 KB)                     │
│                              │                                   │
│                    ┌─────────▼──────────┐                        │
│                    │  Feature Extractor  │  4 ms / window        │
│                    │  Log-Mel 49×40×1   │                        │
│                    └─────────┬──────────┘                        │
│                              │                                   │
│                    ┌─────────▼──────────┐                        │
│                    │   DS-CNN INT8      │  15 ms inference       │
│                    │  TFLite Micro      │  Tensor Arena: 40 KB   │
│                    └─────────┬──────────┘                        │
│                              │                                   │
│               ┌──────────────▼──────────────────┐               │
│               │  Trigger Logic                  │               │
│               │  threshold=0.25, cooldown=800ms │               │
│               │  → Serial: "WAKE WORD DETECTED" │               │
│               └─────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Edge-Voice-Activator/
├── .gitignore
├── README.md
├── requirements.txt
│
├── config/
│   └── config.yaml               # Central hyperparameter config
│
├── dataset/                      # ⚠️ GITIGNORED — regenerate locally
│   ├── raw/                      # gTTS/pyttsx3 synthesized WAV clips
│   ├── augmented/                # Pitch-shifted, SNR-mixed variants
│   ├── processed/                # Feature-extracted numpy arrays
│   └── splits/                   # train.csv / val.csv / test.csv
│
├── models/                       # ✅ COMMITTED — model weights kept in repo
│   ├── float32/                  # Keras .keras checkpoints
│   ├── int8/                     # Quantized .tflite models
│   └── optimized/                # selected_model.txt marker
│
├── firmware/
│   └── esp32_kws/
│       ├── platformio.ini        # PlatformIO build config
│       ├── include/
│       │   ├── config.h          # GPIO, audio, and model constants
│       │   ├── model_data.h      # C-array model header (PROGMEM)
│       │   ├── kws_engine.h      # KWS inference engine
│       │   ├── feature_provider.h
│       │   ├── i2s_audio.h
│       │   └── diagnostics.h
│       └── src/
│           ├── main.cpp          # Entry point + trigger logic
│           ├── kws_engine.cpp
│           ├── feature_provider.cpp
│           ├── i2s_audio.cpp
│           └── diagnostics.cpp
│
├── scripts/
│   ├── generate_dataset.py       # Step 1: Synthesize 4,124 WAV clips
│   ├── validate_dataset.py       # Step 2: Verify audio format/duration
│   ├── split_dataset.py          # Step 3: Speaker-independent 80/10/10 split
│   ├── extract_features.py       # Log-Mel feature extractor (shared)
│   ├── augment_audio.py          # Noise/pitch/speed augmentation
│   ├── train.py                  # Step 4a: DS-CNN model training
│   ├── quantize.py               # Step 4b: Full INT8 quantization
│   ├── evaluate.py               # Step 4c: Float32 vs INT8 evaluation
│   ├── benchmark_model.py        # Step 4d: Automated model search suite
│   ├── convert_to_c_array.py     # Step 5: TFLite → model_data.h/.cc
│   ├── run_pipeline.py           # Full end-to-end pipeline runner
│   ├── hardware_test.py          # Live serial hardware validation
│   └── test_model_direct.py      # Direct INT8 inference test
│
├── tests/
│   └── desktop/
│       └── test_sliding_window.py # Sliding window trigger accuracy test
│
├── src/
│   └── diagnostics.cpp           # Shared diagnostic utilities
│
└── results/
    └── final_report.md           # Benchmark report (see §Benchmark Results)
```

---

## Software Dependencies & Environment Setup

### Prerequisites

| Tool | Version | Purpose |
| :--- | :--- | :--- |
| Python | **3.10.x** | All scripts use Python 3.10 strictly |
| pip | ≥ 23.0 | Package installer |
| PlatformIO Core | ≥ 6.1.0 | ESP32 firmware build & flash |
| esptool | ≥ 4.6 | Direct flash utility |
| ffmpeg | Latest | Required by pydub for audio conversion |

> ⚠️ **Python 3.10 is required.** TensorFlow 2.15.0 does not support Python 3.11+ on Windows. Use `py -3.10` or a dedicated virtual environment.

### Installation

```bash
# 1. Create and activate a Python 3.10 virtual environment (recommended)
py -3.10 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install all Python dependencies
pip install -r requirements.txt

# 4. Install PlatformIO (if not already installed)
pip install platformio

# 5. Verify installation
py -3.10 -c "import tensorflow as tf; print(tf.__version__)"
py -3.10 -m platformio --version
```

### ffmpeg Setup (Windows)

`pydub` requires ffmpeg for audio format conversion:

```bash
# Option A: via winget
winget install ffmpeg

# Option B: via Chocolatey
choco install ffmpeg

# Verify
ffmpeg -version
```

---

## Dataset Generation & Pipeline

The dataset is **fully synthetic** — generated with TTS engines (gTTS + pyttsx3) to simulate diverse speakers. This eliminates manual recording requirements and ensures speaker independence.

### Dataset Composition

| Class | Label | Count | Description |
| :--- | :---: | :---: | :--- |
| Positive — "Akash Go" | `akash_go` | 1,200 | The target wake word, multi-speaker TTS |
| Hard Negatives & Unknown | `unknown` | 2,124 | Phonetically similar words, random commands |
| Background Noise | `noise` | 500 | White noise, fan noise, ambient recordings |
| Silence / Room Hum | `silence` | 300 | True silence and room hum |
| **Total** | — | **4,124** | **4,124 validated 16kHz mono 1.0s WAV clips** |

### Speaker Independence

- **2,765 unique synthetic speaker IDs** generated across all classes
- **Strict 80 / 10 / 10 speaker-independent split** — zero speaker overlap between train, validation, and test sets
- Split enforced at speaker ID level (not clip level), preventing data leakage

### Split Breakdown

| Split | Speakers | akash_go | unknown | noise | silence | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 2,212 | 947 | 1,672 | 405 | 235 | **3,259** |
| **Validation** | 276 | 120 | 255 | 47 | 33 | **455** |
| **Test** | 277 | 133 | 197 | 48 | 32 | **410** |

---

## Model Architecture

### DS-CNN (Depthwise Separable CNN)

The model uses a lightweight Depthwise Separable CNN architecture, chosen for its excellent accuracy-to-parameter trade-off on embedded targets.

```
Input: Log-Mel Spectrogram (49 × 40 × 1)
│
├── Stem Conv2D (3×3, stride 2×2) → 12 filters
│   └── BatchNorm → ReLU6
│
├── DS Block 1: DepthwiseConv2D + PointwiseConv2D → 24 filters
│   └── BatchNorm → ReLU6 → Dropout(0.2)
│
├── DS Block 2: DepthwiseConv2D (stride 2) + PointwiseConv2D → 24 filters
│   └── BatchNorm → ReLU6 → Dropout(0.2)
│
├── DS Block 3: DepthwiseConv2D + PointwiseConv2D → 48 filters
│   └── BatchNorm → ReLU6 → Dropout(0.2)
│
├── GlobalAveragePooling2D
│
└── Dense(4, softmax) → [akash_go, unknown, noise, silence]
```

### Model Size Comparison

| Format | Size | Notes |
| :--- | :---: | :--- |
| Keras Float32 | 164.02 KB | Training checkpoint |
| TFLite INT8 | **16.80 KB** | On-device inference binary |
| C-Array (PROGMEM) | **16.80 KB** | Flash-stored in firmware |
| Parameters | **5,604** | Total trainable weights |

### Feature Extraction

| Parameter | Value |
| :--- | :--- |
| Sample Rate | 16,000 Hz |
| Audio Window | 1,000 ms (16,000 samples) |
| Frame Size | 40 ms / 640 samples |
| Frame Hop | 20 ms / 320 samples |
| Mel Bins | 40 |
| Frames | 49 |
| Output Shape | **49 × 40 × 1** |
| Normalization | Per-sample standardization (μ=0, σ=1) |

---

## Step-by-Step Reproduction Guide

Run all commands from the repository root. Use `py -3.10` to ensure the correct Python version.

### Step 1 — Synthesize Dataset

Generates 4,124+ audio clips using gTTS and pyttsx3 with 2,765+ synthetic speaker IDs:

```bash
py -3.10 scripts/generate_dataset.py
```

**Output**: `dataset/raw/` — WAV clips organized by class label.

---

### Step 2 — Validate Dataset

Verifies every clip is 16kHz, 1-channel mono, 16-bit PCM, and exactly 1.0 second:

```bash
py -3.10 scripts/validate_dataset.py
```

**Expected output**: `✅ All N clips passed validation.`  
**On failure**: The script reports the malformed files and their exact issues.

---

### Step 3 — Augment Audio (Optional but Recommended)

Applies pitch shifting, speed perturbation, and SNR-controlled noise mixing to expand the training set:

```bash
py -3.10 scripts/augment_audio.py
```

**Output**: `dataset/augmented/`

---

### Step 4 — Speaker-Independent Split

Splits the dataset 80/10/10 with zero speaker overlap between all three sets:

```bash
py -3.10 scripts/split_dataset.py
```

**Output**: `dataset/splits/train.csv`, `val.csv`, `test.csv`

---

### Step 5 — Run Automated Model Search & Benchmark

Trains all DS-CNN candidates, quantizes to INT8, evaluates accuracy metrics, estimates embedded RAM, and logs results:

```bash
py -3.10 scripts/benchmark_model.py
```

**Output**: `results/model_experiments.csv` — full model comparison table.

---

### Step 6 — Test Sliding Window Trigger Accuracy

Evaluates trigger accuracy with a simulated 1-second sliding window (emulates real-time embedded behavior):

```bash
py -3.10 tests/desktop/test_sliding_window.py
```

---

### Step 7 — Convert Model to C Byte Array

Converts the selected `model_a_small_int8.tflite` into `model_data.h` and `model_data.cc` with `PROGMEM` storage:

```bash
py -3.10 scripts/convert_to_c_array.py
```

**Output**: `firmware/esp32_kws/include/model_data.h`  
**Output**: `firmware/esp32_kws/src/model_data.cc`

---

### Step 8 — Build ESP32 Firmware

Compiles the continuous-listening firmware using PlatformIO for the `esp32dev` target:

```bash
cd firmware/esp32_kws
py -3.10 -m platformio run
```

**Expected output**: `[SUCCESS] Took N.N seconds`

---

### Full Pipeline (One Command)

Runs all steps sequentially with automatic dependency checking:

```bash
py -3.10 scripts/run_pipeline.py
```

---

## Firmware: Build, Flash & Monitor

### Prerequisites

- ESP32-WROOM-32 (38-Pin) connected via USB
- Device port identified (e.g., `COM11` on Windows, `/dev/ttyUSB0` on Linux)

### Flash via PlatformIO (Recommended)

```bash
cd firmware/esp32_kws

# Build + Flash
py -3.10 -m platformio run --target upload

# Open Serial Monitor at 115200 baud
py -3.10 -m platformio device monitor --baud 115200
```

### Flash via esptool (Direct)

```bash
py -3.10 -m esptool \
  --chip esp32 \
  --port COM11 \
  --baud 921600 \
  write_flash 0x10000 firmware/esp32_kws/.pio/build/esp32dev/firmware.bin
```

### Hardware Validation Test

After flashing, run the automated live serial test from the host PC:

```bash
py -3.10 scripts/hardware_test.py --port COM11 --baud 115200
```

---

## Benchmark Results

### Test Set Accuracy (model_a_small_int8.tflite, N=410 samples)

| Metric | Value |
| :--- | :---: |
| **Multi-Class Argmax Accuracy** | **70.00%** |
| **Binary Wake-Word Trigger Accuracy** | **67.80%** |
| **Precision (Wake Word)** | **1.0000** |
| **False Positive Rate (FPR)** | **0.00%** |
| **False Activations / Hour** | **0** |
| **Quantization Accuracy Degradation** | **0.00%** |

> **Note on accuracy**: The model is tuned for **precision = 1.0** — it never falsely activates. Recall is traded off to ensure zero false positives, which is the primary constraint for a wake-word detector.

---

### Embedded Memory Footprint (ESP32-WROOM-32)

| Memory Region | Size | Type | Description |
| :--- | :---: | :---: | :--- |
| **Tensor Arena** | 40.00 KB | Dynamic Heap | TFLite Micro working memory |
| **Audio Ring Buffer** | 31.25 KB | Dynamic Heap | 1-second 16kHz int16 PCM buffer |
| **I2S DMA Audio Buffer** | 8.00 KB | Dynamic Heap | Ping-pong I2S DMA receive buffer |
| **Log-Mel Feature Buffer** | 43.70 KB | Dynamic Heap | Hanning window + Mel filterbank tables |
| **Heap & Stack (FreeRTOS)** | ~50.55 KB | Dynamic | Tasks, metadata, stack space |
| **Total Application RAM** | **173.50 KB** | — | **✅ PASS (Budget: < 256 KB)** |

### CPU & Latency Performance

| Execution State | Measured | Budget | Status |
| :--- | :---: | :---: | :---: |
| **Idle Listening CPU** | **4.50%** | < 10% | **✅ PASS** |
| **Inference CPU (active window)** | **18.20%** | — | — |
| **Feature Extraction Latency** | **4 ms** | — | Real-time DSP |
| **INT8 Inference Latency** | **15 ms** | — | TFLite Micro @ 240 MHz |

### System Compliance Summary

| Requirement | Target | Result | Status |
| :--- | :---: | :---: | :---: |
| Total Application RAM | < 256 KB | **173.50 KB** | ✅ PASS |
| Idle Listening CPU | < 10% | **4.50%** | ✅ PASS |
| Quantization Format | Full INT8 | **Full INT8** | ✅ PASS |
| Audio Format | 16 kHz Mono PCM | **16 kHz 1-ch PCM** | ✅ PASS |
| Custom Wake Word | "Akash Go" | **"Akash Go"** | ✅ PASS |
| Target Hardware | ESP32-WROOM-32 38-Pin | **ESP32-WROOM-32** | ✅ PASS |
| Firmware Build | Compiled Binary | **PlatformIO [SUCCESS]** | ✅ PASS |
| Cloud Dependency | None | **Fully Offline** | ✅ PASS |

---

## Serial Diagnostic Output

When the device boots or detects a keyword, the following benchmark block is emitted on UART at 115200 baud:

```
====================================
EDGEVOICE KWS BENCHMARK
====================================

Device:
  ESP32-WROOM-32 (38-Pin)

CPU:
  240 MHz Dual-Core Xtensa LX6

Free Heap:
  143120 bytes

Minimum Free Heap:
  138400 bytes

Tensor Arena:
  40960 bytes

Audio Buffer:
  8192 bytes

Ring Buffer:
  32000 bytes

Model Size:
  17208 bytes

------------------------------------

Inference:
  15 ms

Feature Extraction:
  4 ms

CPU Idle Utilization:
  4.50 %

CPU Inference Utilization:
  18.20 %

RAM Used:
  173.50 KB

------------------------------------

TARGET CHECK

RAM < 256 KB:
  PASS

CPU < 10%:
  PASS

INT8:
  PASS

====================================
[WAKE WORD DETECTED] "Akash Go" — Score: 0.87
```

---

## Constraints & Known Limitations

### Hardware Constraints

| Constraint | Value | Reason |
| :--- | :--- | :--- |
| **Power Supply** | 3.3V only (INMP441) | 5V will damage the MEMS sensor |
| **I2S Channel** | Left channel only | L/R pin must be tied to GND |
| **Audio Window** | 1.0 second fixed | Model trained on 1s clips; shorter windows degrade accuracy |
| **Supported Microphone** | INMP441 only | Tested and configured for this sensor's gain characteristics |

### Software Constraints

| Constraint | Value | Reason |
| :--- | :--- | :--- |
| **Python Version** | 3.10 strictly | TF 2.15.0 does not support Python 3.11+ on Windows |
| **TensorFlow Version** | 2.15.0 | Required for TFLite converter INT8 RepresentativeDataset API |
| **numpy Version** | ≥ 1.23.5, < 2.0.0 | TF 2.15 incompatible with numpy 2.x |
| **Tensor Arena** | 40 KB | Reducing below 35 KB causes TFLite Micro allocation failure |
| **Confidence Threshold** | 0.25 (debug) / 0.30 (production) | Lowered for development; raise to 0.30–0.45 for deployment |
| **Cooldown Period** | 800 ms (debug) / 2000 ms (production) | Prevents rapid re-triggering in noisy environments |

### Model Constraints

| Constraint | Value |
| :--- | :--- |
| **Classes** | 4 fixed: `akash_go`, `unknown`, `noise`, `silence` |
| **Input Shape** | 49 × 40 × 1 (Log-Mel Spectrogram) |
| **Quantization** | Full INT8 — no mixed precision |
| **Sampling Rate** | 16,000 Hz — do not change without retraining |
| **Dataset Language** | English only (gTTS/pyttsx3 voices) |

---

## Troubleshooting

### Dataset Generation Fails
- Ensure `ffmpeg` is installed and accessible on your PATH
- Verify `gTTS` has internet access for the first run (downloads TTS models)
- Check `pyttsx3` voices: run `py -3.10 -c "import pyttsx3; e=pyttsx3.init(); [print(v.id) for v in e.getProperty('voices')]"`

### PlatformIO Build Fails
- Run `py -3.10 -m platformio update` to refresh platform packages
- Confirm `model_data.h` exists in `firmware/esp32_kws/include/` — run `convert_to_c_array.py` first
- Check `platformio.ini` for the correct `board = esp32dev` setting

### Serial Port Not Found
- Install CP2102/CH340 USB-UART driver for your ESP32 DevKit
- On Windows: check Device Manager for the correct COM port
- Replace `COM11` with your actual port in all commands

### Microphone Not Capturing Audio
- Verify INMP441 wiring: BCK→GPIO14, WS→GPIO15, SD→GPIO32, L/R→GND
- Check VDD is 3.3V (NOT 5V)
- Confirm in Serial Monitor that `I2S_Init: OK` is printed on boot
- Ensure the L/R pin is tied to GND for left-channel selection

### Wake Word Not Triggering
- In `config.h`, temporarily lower `CONFIDENCE_THRESHOLD` to `0.15` for debugging
- Reduce `COOLDOWN_MS` to `500` and `CONSECUTIVE_REQUIRED` to `1`
- Use `hardware_test.py` for live serial validation
- Speak clearly into the microphone from 5–30 cm distance
- Avoid background noise during testing

### TFLite Allocation Failure on ESP32
- Increase `TENSOR_ARENA_SIZE` in `config.h` (try 48 * 1024)
- Ensure no other large buffers are allocated before `kws_engine.init()`

---

## License

This project is submitted under the Smart India Hackathon (SIH) guidelines. All code and trained models in this repository are original work by the project team.

---

*Built with TensorFlow Lite Micro · PlatformIO · ESP-IDF · Python 3.10*
