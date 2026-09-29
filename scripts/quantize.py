#!/usr/bin/env python3
"""
GroundWatch Full INT8 Quantizer
Converts Float32 Keras model into fully quantized INT8 TFLite model for TFLite Micro.
Enforces INT8 input/output and calibration via representative dataset generator.
"""

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from extract_features import extract_file_features

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATASET_SPLITS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "splits"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
FLOAT_DIR = os.path.join(MODELS_DIR, "float32")
INT8_DIR = os.path.join(MODELS_DIR, "int8")
os.makedirs(INT8_DIR, exist_ok=True)

def representative_dataset_gen():
    """Generates representative samples from training split for INT8 calibration."""
    df = pd.read_csv(os.path.join(DATASET_SPLITS, "train.csv"))
    # Select 200 calibration samples randomly
    sample_rows = df.sample(n=min(200, len(df)), random_state=42)
    for idx, row in sample_rows.iterrows():
        feat = extract_file_features(row["filepath"])
        # Yield as batch of 1 with float32
        yield [np.expand_dims(feat, axis=0).astype(np.float32)]

def quantize_to_int8(model_name="model_b_smaller"):
    keras_path = os.path.join(FLOAT_DIR, f"{model_name}.keras")
    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"Float32 model not found: {keras_path}")
        
    print(f"\n--- Quantizing {model_name} to Full INT8 ---")
    model = tf.keras.models.load_model(keras_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Enforce FULL INT8 (weights + activations + input + output)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_int8_model = converter.convert()
    
    int8_path = os.path.join(INT8_DIR, f"{model_name}_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(tflite_int8_model)
        
    file_size_kb = len(tflite_int8_model) / 1024.0
    print(f"Quantization Successful!")
    print(f"  INT8 TFLite Model Path: {int8_path}")
    print(f"  INT8 Model File Size:  {file_size_kb:.2f} KB")
    
    # Test loading in TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=int8_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Interpreter Input Scale:  {input_details[0]['quantization'][0]}")
    print(f"  Interpreter Input ZeroPt: {input_details[0]['quantization'][1]}")
    print(f"  Interpreter Input Dtype:  {input_details[0]['dtype']}")
    print(f"  Interpreter Output Dtype: {output_details[0]['dtype']}")
    
    return int8_path, len(tflite_int8_model)

if __name__ == "__main__":
    quantize_to_int8("model_b_smaller")
