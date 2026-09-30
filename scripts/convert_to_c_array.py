#!/usr/bin/env python3
"""
VoiceEdge TFLite-to-C Array Converter (Flash PROGMEM Optimized)
Converts quantized .tflite INT8 model into alignas(16) PROGMEM static byte array C/C++ files.
Output header and source files are placed directly in the ESP32 firmware src directory.
"""

import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
INT8_DIR = os.path.join(MODELS_DIR, "int8")
OPT_DIR = os.path.join(MODELS_DIR, "optimized")

FIRMWARE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "firmware", "esp32_kws"))
FW_SRC_DIR = os.path.join(FIRMWARE_DIR, "src")
FW_INC_DIR = os.path.join(FIRMWARE_DIR, "include")
os.makedirs(FW_SRC_DIR, exist_ok=True)
os.makedirs(FW_INC_DIR, exist_ok=True)

def convert_tflite_to_c(model_name="model_a_small"):
    tflite_path = os.path.join(INT8_DIR, f"{model_name}_int8.tflite")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
        
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
        
    model_len = len(model_bytes)
    print(f"Converting '{model_name}_int8.tflite' ({model_len} bytes) into C byte array (PROGMEM)...")
    
    header_content = f"""/* VoiceEdge Auto-generated Model Header */
#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <cstdint>
#include <pgmspace.h>

alignas(16) extern const unsigned char g_model[{model_len}];
extern const int g_model_len;

#endif  // MODEL_DATA_H_
"""
    
    header_path = os.path.join(FW_INC_DIR, "model_data.h")
    with open(header_path, "w") as f:
        f.write(header_content)
        
    bytes_per_line = 12
    hex_lines = []
    for i in range(0, model_len, bytes_per_line):
        chunk = model_bytes[i:i+bytes_per_line]
        hex_str = ", ".join([f"0x{b:02x}" for b in chunk])
        hex_lines.append("  " + hex_str)
        
    array_body = ",\n".join(hex_lines)
    
    source_content = f"""/* VoiceEdge Auto-generated Model Source Array (Flash Storage) */
#include "model_data.h"

const int g_model_len = {model_len};

alignas(16) const unsigned char g_model[{model_len}] PROGMEM = {{
{array_body}
}};
"""

    source_path = os.path.join(FW_SRC_DIR, "model_data.cc")
    with open(source_path, "w") as f:
        f.write(source_content)
        
    print(f"Model Conversion Complete!")
    print(f"  Header File: {header_path}")
    print(f"  Source File: {source_path}")
    print(f"  Total Model Flash Size: {model_len / 1024.0:.2f} KB")

def main():
    sel_file = os.path.join(OPT_DIR, "selected_model.txt")
    if os.path.exists(sel_file):
        with open(sel_file, "r") as f:
            model_name = f.read().strip()
    else:
        model_name = "model_a_small"
        
    convert_tflite_to_c(model_name)

if __name__ == "__main__":
    main()
