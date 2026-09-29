#!/usr/bin/env python3
"""
EdgeVoice DS-CNN Model Trainer
Builds and trains Depthwise Separable CNN architectures for Keyword Spotting.
Designed specifically for high accuracy with minimal memory footprint and INT8 compatibility.
"""

import os
import random
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from extract_features import extract_file_features

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATASET_SPLITS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "splits"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
FLOAT_DIR = os.path.join(MODELS_DIR, "float32")
os.makedirs(FLOAT_DIR, exist_ok=True)

LABEL_MAP = {"akash_go": 0, "unknown": 1, "noise": 2, "silence": 3}
NUM_CLASSES = len(LABEL_MAP)
INPUT_SHAPE = tuple(config["features"]["input_shape"])  # (49, 40, 1)

def load_split_data(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    for idx, row in df.iterrows():
        feat = extract_file_features(row["filepath"])
        X.append(feat)
        y.append(LABEL_MAP[row["label"]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def build_dscnn_model(conv_filters=12, ds_filters=[24, 24, 48], ds_strides=[1, 2, 1], dropout=0.2):
    model = models.Sequential(name="DS_CNN_KWS")
    model.add(layers.Input(shape=INPUT_SHAPE))
    
    # Stem Conv2D
    model.add(layers.Conv2D(conv_filters, (3, 3), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU(max_value=6.0))
    
    # Depthwise Separable Conv Blocks
    for i, (f, s) in enumerate(zip(ds_filters, ds_strides)):
        model.add(layers.DepthwiseConv2D((3, 3), strides=(s, s), padding="same", use_bias=False, name=f"ds_conv_{i+1}_dw"))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU(max_value=6.0))
        
        model.add(layers.Conv2D(f, (1, 1), strides=(1, 1), padding="same", use_bias=False, name=f"ds_conv_{i+1}_pw"))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU(max_value=6.0))
        
        if dropout > 0:
            model.add(layers.Dropout(dropout))
            
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="output_dense"))
    
    return model

def train_model(model_name="model_b_smaller", params=None):
    if params is None:
        params = config["model_search"]["models"]["model_b_smaller"]
        
    print(f"\n--- Training {model_name} ---")
    model = build_dscnn_model(
        conv_filters=params["conv_filters"],
        ds_filters=params["ds_filters"],
        ds_strides=params["ds_strides"],
        dropout=params["dropout"]
    )
    
    model.summary()
    
    print("Loading train dataset...")
    X_train, y_train = load_split_data(os.path.join(DATASET_SPLITS, "train.csv"))
    print("Loading val dataset...")
    X_val, y_val = load_split_data(os.path.join(DATASET_SPLITS, "val.csv"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    save_path = os.path.join(FLOAT_DIR, f"{model_name}.keras")
    
    cb_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=config["training"]["early_stopping_patience"], restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        callbacks=cb_list,
        verbose=1
    )
    
    print(f"Model saved to: {save_path}")
    return model, history, (X_train, y_train), (X_val, y_val)

if __name__ == "__main__":
    train_model("model_b_smaller")
