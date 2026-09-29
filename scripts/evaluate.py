#!/usr/bin/env python3
"""
GroundWatch Model Evaluator
Evaluates Float32 Keras and INT8 TFLite models on held-out test split.
Computes Accuracy, Precision, Recall, F1, Confusion Matrix, FPR, and FNR.
"""

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from extract_features import extract_file_features

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATASET_SPLITS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "splits"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
FLOAT_DIR = os.path.join(MODELS_DIR, "float32")
INT8_DIR = os.path.join(MODELS_DIR, "int8")

LABEL_MAP = {"akash_go": 0, "unknown": 1, "noise": 2, "silence": 3}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_test_dataset():
    df = pd.read_csv(os.path.join(DATASET_SPLITS, "test.csv"))
    X = []
    y = []
    for idx, row in df.iterrows():
        feat = extract_file_features(row["filepath"])
        X.append(feat)
        y.append(LABEL_MAP[row["label"]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def evaluate_float32(model_path, X_test, y_test):
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2, 3])
    
    # False positive rate for 'akash_go' (class 0)
    # FP: non-0 predicted as 0. TN: non-0 predicted as non-0.
    fp = np.sum((y_test != 0) & (preds == 0))
    tn = np.sum((y_test != 0) & (preds != 0))
    fn = np.sum((y_test == 0) & (preds != 0))
    tp = np.sum((y_test == 0) & (preds == 0))
    
    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))
    
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": cm.tolist()
    }

def evaluate_int8(tflite_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    in_scale, in_zero_pt = input_details[0]['quantization']
    out_scale, out_zero_pt = output_details[0]['quantization']
    
    preds = []
    
    for i in range(len(X_test)):
        sample = X_test[i]
        # Quantize float sample to int8 input
        int8_input = np.clip(np.round(sample / in_scale) + in_zero_pt, -128, 127).astype(np.int8)
        int8_input = np.expand_dims(int8_input, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], int8_input)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Dequantize or argmax directly
        pred_label = np.argmax(output_data)
        preds.append(pred_label)
        
    preds = np.array(preds)
    
    acc = accuracy_score(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2, 3])
    
    fp = np.sum((y_test != 0) & (preds == 0))
    tn = np.sum((y_test != 0) & (preds != 0))
    fn = np.sum((y_test == 0) & (preds != 0))
    tp = np.sum((y_test == 0) & (preds == 0))
    
    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))
    
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": cm.tolist()
    }

def evaluate_model_pair(model_name="model_b_smaller"):
    X_test, y_test = load_test_dataset()
    
    float_path = os.path.join(FLOAT_DIR, f"{model_name}.keras")
    int8_path = os.path.join(INT8_DIR, f"{model_name}_int8.tflite")
    
    res_float = evaluate_float32(float_path, X_test, y_test)
    res_int8 = evaluate_int8(int8_path, X_test, y_test)
    
    print(f"\n==========================================")
    print(f"EVALUATION SUMMARY: {model_name}")
    print(f"==========================================")
    print(f"Metric       | Float32   | INT8 Quantized")
    print(f"------------------------------------------")
    print(f"Accuracy     | {res_float['accuracy']*100:.2f}%    | {res_int8['accuracy']*100:.2f}%")
    print(f"Precision    | {res_float['precision']:.4f}   | {res_int8['precision']:.4f}")
    print(f"Recall       | {res_float['recall']:.4f}   | {res_int8['recall']:.4f}")
    print(f"F1 Score     | {res_float['f1']:.4f}   | {res_int8['f1']:.4f}")
    print(f"False Pos Rate| {res_float['fpr']*100:.2f}%     | {res_int8['fpr']*100:.2f}%")
    print(f"False Neg Rate| {res_float['fnr']*100:.2f}%     | {res_int8['fnr']*100:.2f}%")
    
    deg = (res_float['accuracy'] - res_int8['accuracy']) * 100.0
    print(f"\nQuantization Accuracy Degradation: {deg:.2f}%")
    if deg > 2.0:
        print("WARNING: Significant quantization degradation detected (>2.0%).")
    else:
        print("PASS: INT8 model retains desktop Float32 performance.")
        
    return res_float, res_int8

if __name__ == "__main__":
    evaluate_model_pair("model_b_smaller")
