#!/usr/bin/env python3
"""
GroundWatch – Master Pipeline Runner
=====================================
Runs the complete ML pipeline in a single command:

  Step 1 : Wipe old dataset, generate new real-TTS dataset (gTTS)
  Step 2 : Split dataset (speaker-independent train/val/test)
  Step 3 : Train DS-CNN model (Float32)
  Step 4 : Quantize to Full INT8 TFLite
  Step 5 : Evaluate Float32 + INT8 on held-out test set
  Step 6 : Convert INT8 TFLite → C byte array (model_data.h / model_data.cc)
  Step 7 : Print final summary and next steps (flash firmware)

Usage:
  py -3.10 scripts/run_pipeline.py

Individual steps can also be skipped by passing --start-from N (1-indexed).
"""

import os
import sys
import time
import shutil
import argparse

# Force UTF-8 output on Windows CP1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

# Make scripts importable
sys.path.insert(0, SCRIPTS_DIR)


def banner(step: int, title: str):
    print("\n" + "=" * 65)
    print(f"  STEP {step}: {title}")
    print("=" * 65)


def step1_generate(wipe: bool = True):
    banner(1, "Generate Real-TTS Dataset")

    if wipe:
        raw_dir = os.path.join(PROJECT_ROOT, "dataset", "raw")
        for sub in ["positive", "negative", "unknown", "noise"]:
            sub_path = os.path.join(raw_dir, sub)
            if os.path.isdir(sub_path):
                shutil.rmtree(sub_path)
                print(f"  Wiped: {sub_path}")

        meta = os.path.join(raw_dir, "metadata.csv")
        if os.path.exists(meta):
            os.remove(meta)
            print(f"  Wiped: metadata.csv")

        splits_dir = os.path.join(PROJECT_ROOT, "dataset", "splits")
        if os.path.isdir(splits_dir):
            shutil.rmtree(splits_dir)
            print(f"  Wiped: splits/")

    import generate_dataset
    generate_dataset.main()


def step2_split():
    banner(2, "Speaker-Independent Dataset Split")
    import split_dataset
    split_dataset.main()


def step3_train():
    banner(3, "Train DS-CNN Float32 Model")
    import train
    model, history, _, _ = train.train_model("model_b_smaller")
    return model


def step4_quantize():
    banner(4, "Quantize Float32 → Full INT8 TFLite")
    import quantize
    int8_path, size_bytes = quantize.quantize_to_int8("model_b_smaller")
    print(f"\n  INT8 model: {int8_path}  ({size_bytes/1024:.2f} KB)")
    return int8_path


def step5_evaluate():
    banner(5, "Evaluate Float32 and INT8 Models")
    import evaluate
    res_float, res_int8 = evaluate.evaluate_model_pair("model_b_smaller")

    # Save selected model marker
    opt_dir = os.path.join(PROJECT_ROOT, "models", "optimized")
    os.makedirs(opt_dir, exist_ok=True)
    with open(os.path.join(opt_dir, "selected_model.txt"), "w") as f:
        f.write("model_b_smaller")

    return res_float, res_int8


def step6_convert_c_array():
    banner(6, "Convert INT8 TFLite → Firmware C Array")
    import convert_to_c_array
    convert_to_c_array.convert_tflite_to_c("model_b_smaller")


def step7_summary(res_float=None, res_int8=None):
    banner(7, "Pipeline Complete — Summary")

    fw_src = os.path.join(PROJECT_ROOT, "firmware", "esp32_kws", "src", "model_data.cc")
    fw_inc = os.path.join(PROJECT_ROOT, "firmware", "esp32_kws", "include", "model_data.h")
    int8_path = os.path.join(PROJECT_ROOT, "models", "int8", "model_b_smaller_int8.tflite")

    model_kb = os.path.getsize(int8_path) / 1024.0 if os.path.exists(int8_path) else 0

    print()
    print("  +----------------------------------------------------------+")
    print("  | GroundWatch KWS Pipeline -- Results                      |")
    print("  +----------------------------------------------------------+")

    if res_float and res_int8:
        print("  | Float32 Accuracy:  %6.2f%%                              |" % (res_float['accuracy']*100))
        print("  | INT8    Accuracy:  %6.2f%%                              |" % (res_int8['accuracy']*100))
        print("  | Quant Degradation: %+.2f%%                              |" % ((res_float['accuracy']-res_int8['accuracy'])*100))
        print("  | Wake-word FPR:     %6.2f%%  (False Positive Rate)      |" % (res_int8['fpr']*100))
        print("  | Wake-word FNR:     %6.2f%%  (False Negative Rate)      |" % (res_int8['fnr']*100))

    print("  | INT8 Model Size:   %6.2f KB                              |" % model_kb)
    print("  +----------------------------------------------------------+")
    print("  | Firmware files updated:                                  |")
    print("  |   %s |" % fw_inc[-55:].ljust(55))
    print("  |   %s |" % fw_src[-55:].ljust(55))
    print("  +----------------------------------------------------------+")
    print("  | NEXT STEP: Rebuild + flash firmware                      |")
    print("  |   cd firmware\\esp32_kws                                  |")
    print("  |   pio run --target upload                                |")
    print("  +----------------------------------------------------------+")
    print()


def main():
    parser = argparse.ArgumentParser(description="GroundWatch Pipeline Runner")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from step N (1=generate, 2=split, 3=train, …)")
    parser.add_argument("--no-wipe", action="store_true",
                        help="Skip wiping old dataset in step 1")
    args = parser.parse_args()

    start = args.start_from
    t_total = time.time()

    res_float = res_int8 = None

    if start <= 1:
        step1_generate(wipe=not args.no_wipe)
    if start <= 2:
        step2_split()
    if start <= 3:
        step3_train()
    if start <= 4:
        step4_quantize()
    if start <= 5:
        res_float, res_int8 = step5_evaluate()
    if start <= 6:
        step6_convert_c_array()

    step7_summary(res_float, res_int8)

    elapsed = time.time() - t_total
    print(f"  Total pipeline time: {elapsed/60:.1f} minutes\n")


if __name__ == "__main__":
    main()
