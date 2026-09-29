#!/usr/bin/env python3
"""
GroundWatch Direct Model Test
==============================
Runs the INT8 TFLite model DIRECTLY on WAV files in Python.
No hardware, no speaker, no mic involved.

This is the ground truth test:
  - If positive WAVs score high   -> model is good, audio chain is the problem
  - If positive WAVs score low    -> model itself is broken, need retrain

Usage:
  py -3.10 scripts/test_model_direct.py
  py -3.10 scripts/test_model_direct.py --file path/to/sample.wav
  py -3.10 scripts/test_model_direct.py --all-positives
"""

import os, sys, argparse, random
import numpy as np

# Add scripts dir to path for extract_features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import extract_file_features

import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INT8_MODEL   = os.path.join(PROJECT_ROOT, "models", "int8", "model_b_smaller_int8.tflite")
POS_DIR      = os.path.join(PROJECT_ROOT, "dataset", "raw", "positive")
NEG_DIR      = os.path.join(PROJECT_ROOT, "dataset", "raw", "negative")
UNK_DIR      = os.path.join(PROJECT_ROOT, "dataset", "raw", "unknown")

LABEL_NAMES  = ["akash_go", "unknown", "noise", "silence"]
THRESHOLD    = 0.25


def load_interpreter():
    if not os.path.exists(INT8_MODEL):
        print(f"ERROR: Model not found: {INT8_MODEL}")
        sys.exit(1)
    interp = tf.lite.Interpreter(model_path=INT8_MODEL)
    interp.allocate_tensors()
    return interp


def run_inference(interp, wav_path):
    """Run INT8 model on a WAV file. Returns (predicted_label, confidence, all_probs)."""
    inp  = interp.get_input_details()[0]
    out  = interp.get_output_details()[0]
    in_scale, in_zp   = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    feat = extract_file_features(wav_path)                          # (49,40,1) float32
    int8_in = np.clip(np.round(feat / in_scale) + in_zp,
                      -128, 127).astype(np.int8)
    int8_in = np.expand_dims(int8_in, axis=0)                      # (1,49,40,1)

    interp.set_tensor(inp["index"], int8_in)
    interp.invoke()

    raw_out  = interp.get_tensor(out["index"])[0]                  # int8[4]
    probs    = (raw_out.astype(np.float32) - out_zp) * out_scale   # dequantize
    probs    = np.exp(probs - np.max(probs))                       # softmax
    probs   /= probs.sum()

    pred_idx = int(np.argmax(probs))
    return LABEL_NAMES[pred_idx], float(probs[pred_idx]), probs


def print_bar(label, prob, width=40):
    filled = int(prob * width)
    bar    = "|" * filled + "." * (width - filled)
    mark   = " ** TRIGGER **" if label == "akash_go" and prob >= THRESHOLD else ""
    print(f"  {label:<12} [{bar}] {prob:.4f}{mark}")


def test_single(interp, wav_path, verbose=True):
    pred, conf, probs = run_inference(interp, wav_path)
    if verbose:
        fname = os.path.basename(wav_path)
        triggered = pred == "akash_go" and conf >= THRESHOLD
        status = "DETECTED" if triggered else "missed"
        print(f"\n  File: {fname}")
        print(f"  Result: [{status}]  pred={pred}  conf={conf:.4f}")
        for i, name in enumerate(LABEL_NAMES):
            print_bar(name, float(probs[i]))
    return pred, conf


def test_batch(interp, wav_dir, label, n=20):
    """Run on up to n random WAV files from a directory."""
    files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    if not files:
        print(f"  No WAV files in {wav_dir}")
        return
    sample = random.sample(files, min(n, len(files)))

    correct = 0
    scores  = []
    for fname in sample:
        fpath = os.path.join(wav_dir, fname)
        pred, conf, _ = run_inference(interp, fpath)
        scores.append(conf if label == "akash_go" else (1 - conf))
        hit = (pred == label) if label != "unknown" else (pred != "akash_go")
        if hit: correct += 1
        tick = "OK" if hit else "XX"
        print(f"  [{tick}] {fname:35s}  pred={pred:<12}  conf={conf:.4f}")

    acc = correct / len(sample) * 100
    avg_score = sum(scores) / len(scores)
    print(f"\n  Accuracy on {label}: {correct}/{len(sample)} = {acc:.1f}%")
    print(f"  Avg target-class score: {avg_score:.4f}")
    return acc, avg_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",          help="Test a single WAV file")
    parser.add_argument("--all-positives", action="store_true",
                        help="Run all positive samples and report accuracy")
    parser.add_argument("--negatives",     action="store_true",
                        help="Run negative samples and check FPR")
    parser.add_argument("-n", type=int, default=20,
                        help="Number of samples per class to test (default 20)")
    args = parser.parse_args()

    print("\n  GroundWatch Direct Model Test (No Hardware Needed)")
    print(f"  Model: {os.path.basename(INT8_MODEL)}")
    print(f"  Threshold: {THRESHOLD}\n")

    interp = load_interpreter()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"  Input:  scale={inp_det['quantization'][0]:.4f}  zero_pt={inp_det['quantization'][1]}")
    print(f"  Output: scale={out_det['quantization'][0]:.4f}  zero_pt={out_det['quantization'][1]}")
    print(f"  Input dtype: {inp_det['dtype']}  Output dtype: {out_det['dtype']}")

    if args.file:
        # Single file test
        if not os.path.exists(args.file):
            print(f"\nERROR: File not found: {args.file}")
            sys.exit(1)
        test_single(interp, args.file)
        return

    # ── Default: test 20 positives + 20 negatives ──────────────────────────
    if not args.negatives or args.all_positives:
        print("\n" + "="*60)
        print("POSITIVE SAMPLES  (should predict 'akash_go' with high confidence)")
        print("="*60)
        if os.path.isdir(POS_DIR):
            pos_acc, pos_avg = test_batch(interp, POS_DIR, "akash_go", args.n)
        else:
            print(f"  Positive dir not found: {POS_DIR}")
            pos_acc, pos_avg = 0, 0

    if not args.all_positives or args.negatives:
        print("\n" + "="*60)
        print("NEGATIVE SAMPLES  (should NOT predict 'akash_go')")
        print("="*60)
        neg_dir = NEG_DIR if os.path.isdir(NEG_DIR) and os.listdir(NEG_DIR) else UNK_DIR
        if os.path.isdir(neg_dir):
            neg_acc, neg_avg = test_batch(interp, neg_dir, "unknown", args.n)
        else:
            print(f"  Negative dir not found: {neg_dir}")

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    if 'pos_acc' in dir():
        if pos_acc >= 90:
            print("  Positive accuracy %.1f%%  -> Model is GOOD on training data" % pos_acc)
            print("  The problem is in the AUDIO CHAIN (speaker -> mic distortion)")
            print("  Tips to fix hardware testing:")
            print("    1. Hold speaker <2cm from mic")
            print("    2. Use phone speaker (not laptop) - better frequency response")
            print("    3. Play WAV at MAXIMUM volume")
            print("    4. Use test --play below to automate playback")
        elif pos_acc >= 70:
            print("  Positive accuracy %.1f%%  -> Model is WEAK" % pos_acc)
            print("  Recommendation: Retrain with augmented data or more epochs")
        else:
            print("  Positive accuracy %.1f%%  -> Model FAILS on its own training data" % pos_acc)
            print("  This is a critical failure. Check:")
            print("    - Feature extraction mismatch (Python vs firmware DSP)")
            print("    - Model quantization issue")
            print("    - INT8 scale/zero-point mismatch")

    print("\n  To test a specific WAV file:")
    pos_example = os.path.join(POS_DIR, "pos_akash_go_0000.wav") if os.path.isdir(POS_DIR) else "path/to/file.wav"
    print(f"    py -3.10 scripts/test_model_direct.py --file \"{pos_example}\"")


if __name__ == "__main__":
    main()
