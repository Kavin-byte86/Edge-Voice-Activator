#!/usr/bin/env python3
"""
EdgeVoice Desktop Continuous Stream Simulator & Accuracy Evaluation Suite
Evaluates continuous 16kHz audio streams and measures TP, TN, FP, FN, Accuracy, Precision, Recall, F1, and False Activations/Hour.
"""

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io.wavfile as wav
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
from extract_features import compute_log_mel_spectrogram

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml"))
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
SEL_FILE = os.path.join(MODELS_DIR, "optimized", "selected_model.txt")
if os.path.exists(SEL_FILE):
    with open(SEL_FILE, "r") as f:
        model_name = f.read().strip()
else:
    model_name = "model_a_small"

INT8_MODEL_PATH = os.path.join(MODELS_DIR, "int8", f"{model_name}_int8.tflite")
DATASET_SPLITS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "splits"))

CONFIDENCE_THRESHOLD = config["trigger_logic"]["confidence_threshold"]

def load_tflite_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_single_window(interpreter, float_16000_samples):
    spec = compute_log_mel_spectrogram(float_16000_samples)
    spec_4d = np.expand_dims(np.expand_dims(spec, axis=-1), axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    in_scale, in_zero_pt = input_details[0]['quantization']
    out_scale, out_zero_pt = output_details[0]['quantization']
    
    int8_input = np.clip(np.round(spec_4d / in_scale) + in_zero_pt, -128, 127).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], int8_input)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    logits = (output_data.astype(np.float32) - out_zero_pt) * out_scale
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs, np.argmax(output_data)

def main():
    print(f"=== EdgeVoice Desktop Continuous Stream Simulator ({model_name}) ===")
    if not os.path.exists(INT8_MODEL_PATH):
        raise FileNotFoundError(f"INT8 TFLite model not found: {INT8_MODEL_PATH}")
        
    interpreter = load_tflite_interpreter(INT8_MODEL_PATH)
    
    test_df = pd.read_csv(os.path.join(DATASET_SPLITS, "test.csv"))
    print(f"Loaded {len(test_df)} test stream samples.")
    
    tp, tn, fp, fn = 0, 0, 0, 0
    total_non_keyword_seconds = 0.0
    
    pos_confidences = []
    correct_argmax = 0
    
    for idx, row in test_df.iterrows():
        sr, audio = wav.read(row["filepath"])
        float_audio = audio.astype(np.float32) / 32768.0
        
        probs, argmax_label = run_single_window(interpreter, float_audio)
        akash_go_prob = probs[0]
        
        target_label = 0 if row["label"] == "akash_go" else (1 if row["label"] == "unknown" else (2 if row["label"] == "noise" else 3))
        if argmax_label == target_label:
            correct_argmax += 1
            
        is_positive = (row["label"] == "akash_go")
        if is_positive:
            pos_confidences.append(akash_go_prob)
            
        predicted_trigger = (akash_go_prob >= 0.35)
        
        if is_positive:
            if predicted_trigger:
                tp += 1
            else:
                fn += 1
        else:
            total_non_keyword_seconds += 1.0
            if predicted_trigger:
                fp += 1
            else:
                tn += 1
                
    overall_accuracy = (correct_argmax / len(test_df)) * 100.0
    binary_accuracy = (tp + tn) / max(1, (tp + tn + fp + fn)) * 100.0
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-6, (precision + recall))
    fpr = fp / max(1, (fp + tn))
    
    total_hours = total_non_keyword_seconds / 3600.0
    false_activations_per_hour = fp / max(1e-6, total_hours)
    
    print("\n==================================================")
    print("EDGEVOICE ACCURACY & PERFORMANCE METRICS")
    print("==================================================")
    print(f"Model Evaluated:               {model_name}_int8.tflite")
    print(f"Multi-Class Argmax Accuracy:   {overall_accuracy:.2f}%")
    print(f"Binary Trigger Accuracy:       {binary_accuracy:.2f}%")
    print(f"True Positives  (TP):          {tp}")
    print(f"True Negatives  (TN):          {tn}")
    print(f"False Positives (FP):          {fp}")
    print(f"False Negatives (FN):          {fn}")
    print(f"--------------------------------------------------")
    print(f"Precision:                     {precision:.4f}")
    print(f"Recall:                        {recall:.4f}")
    print(f"F1 Score:                      {f1:.4f}")
    print(f"False Positive Rate (FPR):     {fpr*100:.2f}%")
    print(f"False Activations / Hour:      {false_activations_per_hour:.2f}")
    print("==================================================")

if __name__ == "__main__":
    main()
