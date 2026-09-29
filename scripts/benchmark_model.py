#!/usr/bin/env python3
"""
GroundWatch Automated Model Search & Benchmark Suite
Trains, quantizes, and evaluates progressive DS-CNN candidates.
Logs all results to results/model_experiments.csv and picks the optimal embedded model.
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from train import train_model
from quantize import quantize_to_int8
from evaluate import evaluate_model_pair

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
EXP_CSV = os.path.join(RESULTS_DIR, "model_experiments.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

def count_macs(model):
    """Estimates Multiply-Accumulate (MAC) operations for DS-CNN."""
    total_macs = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # output_h * output_w * in_channels * out_channels * kernel_h * kernel_w
            out_shape = layer.output_shape
            h, w = out_shape[1], out_shape[2]
            k_h, k_w = layer.kernel_size
            c_in = layer.input_shape[-1]
            c_out = layer.filters
            macs = h * w * c_in * c_out * k_h * k_w
            total_macs += macs
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            out_shape = layer.output_shape
            h, w = out_shape[1], out_shape[2]
            k_h, k_w = layer.kernel_size
            c_in = layer.input_shape[-1]
            macs = h * w * c_in * k_h * k_w
            total_macs += macs
        elif isinstance(layer, tf.keras.layers.Dense):
            macs = layer.input_shape[-1] * layer.units
            total_macs += macs
    return int(total_macs)

def estimate_tensor_arena(model, input_shape=(49, 40, 1)):
    """Estimates TFLite Micro Tensor Arena memory footprint (in bytes)."""
    # Max memory required for two working activation buffers during layer execution
    max_scratch = 0
    prev_bytes = np.prod(input_shape) * 1  # INT8 = 1 byte
    
    for layer in model.layers:
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        out_bytes = np.prod([dim for dim in out_shape[1:] if dim is not None]) * 1
        scratch = prev_bytes + out_bytes
        if scratch > max_scratch:
            max_scratch = scratch
        prev_bytes = out_bytes
        
    # Add safety overhead for TFLM interpreter structs & tensor heads (~16 KB)
    arena_estimate_bytes = max_scratch + (16 * 1024)
    return int(arena_estimate_bytes)

def benchmark_single_model(model_name, params, exp_id):
    print(f"\n==================================================")
    print(f"BENCHMARKING EXPERIMENT #{exp_id}: {model_name}")
    print(f"==================================================")
    
    # 1. Train
    model, history, train_data, val_data = train_model(model_name, params)
    
    params_count = model.count_params()
    estimated_macs = count_macs(model)
    tensor_arena_bytes = estimate_tensor_arena(model)
    
    float_path = os.path.join(os.path.dirname(__file__), "..", "models", "float32", f"{model_name}.keras")
    float_size_bytes = os.path.getsize(float_path)
    
    # 2. Quantize
    int8_path, int8_size_bytes = quantize_to_int8(model_name)
    
    # 3. Evaluate
    res_float, res_int8 = evaluate_model_pair(model_name)
    
    # 4. Benchmark TFLite Interpreter Latency
    interpreter = tf.lite.Interpreter(model_path=int8_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    dummy_input = np.zeros(input_details[0]['shape'], dtype=np.int8)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
    t_starts = []
    for _ in range(50):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        t1 = time.perf_counter()
        t_starts.append((t1 - t0) * 1000.0)
        
    avg_latency_ms = float(np.mean(t_starts))
    
    # Check compliance with budget
    estimated_ram_kb = (tensor_arena_bytes + 24 * 1024) / 1024.0  # Tensor Arena + Ring Buffer
    pass_status = "PASS" if (res_int8["accuracy"] >= 0.85 and estimated_ram_kb < 256.0) else "FAIL"
    
    exp_record = {
        "experiment_id": exp_id,
        "model_name": model_name,
        "parameters": params_count,
        "float32_size_kb": float_size_bytes / 1024.0,
        "int8_size_kb": int8_size_bytes / 1024.0,
        "accuracy": res_int8["accuracy"],
        "precision": res_int8["precision"],
        "recall": res_int8["recall"],
        "f1": res_int8["f1"],
        "false_positive_rate": res_int8["fpr"],
        "false_negative_rate": res_int8["fnr"],
        "estimated_arena_ram_kb": tensor_arena_bytes / 1024.0,
        "estimated_total_ram_kb": estimated_ram_kb,
        "estimated_macs": estimated_macs,
        "inference_latency_ms": avg_latency_ms,
        "status": pass_status
    }
    
    return exp_record

def main():
    print("=== GroundWatch Model Search Optimization Pipeline Starting ===")
    search_models = config["model_search"]["models"]
    
    experiments = []
    
    for i, (model_name, params) in enumerate(search_models.items(), start=1):
        rec = benchmark_single_model(model_name, params, i)
        experiments.append(rec)
        
    df_exp = pd.DataFrame(experiments)
    df_exp.to_csv(EXP_CSV, index=False)
    
    print(f"\n==================================================")
    print(f"MODEL EXPERIMENT LOG COMPLETE: {EXP_CSV}")
    print(f"==================================================")
    print(df_exp[["model_name", "parameters", "int8_size_kb", "accuracy", "f1", "false_positive_rate", "estimated_total_ram_kb", "status"]])
    
    # Pick optimal passing model with best F1 score
    valid_models = df_exp[df_exp["status"] == "PASS"]
    if len(valid_models) == 0:
        best_model_name = df_exp.sort_values(by="f1", ascending=False).iloc[0]["model_name"]
    else:
        best_model_name = valid_models.sort_values(by="f1", ascending=False).iloc[0]["model_name"]
        
    print(f"\nOptimal Model Selected for ESP32 Deployment: {best_model_name}")
    
    # Save optimized model marker
    opt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "optimized"))
    os.makedirs(opt_dir, exist_ok=True)
    with open(os.path.join(opt_dir, "selected_model.txt"), "w") as f:
        f.write(best_model_name)

if __name__ == "__main__":
    main()
