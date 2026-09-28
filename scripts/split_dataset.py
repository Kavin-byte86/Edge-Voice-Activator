#!/usr/bin/env python3
"""
GroundWatch Speaker-Independent Dataset Splitter
Splits raw dataset into train/val/test sets without speaker leakage.
"""

import os
import random
import yaml
import pandas as pd
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SEED = config["training"]["seed"]
random.seed(SEED)
np.random.seed(SEED)

DATASET_RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "raw"))
DATASET_SPLITS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "splits"))
os.makedirs(DATASET_SPLITS, exist_ok=True)

META_PATH = os.path.join(DATASET_RAW, "metadata.csv")

def main():
    print("=== GroundWatch Speaker-Independent Splitter Starting ===")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}. Run generate_dataset.py first.")
        
    df = pd.read_csv(META_PATH)
    print(f"Loaded {len(df)} metadata records.")
    
    # Separate speech samples (which have speaker_ids) from environment samples (noise/silence)
    speech_df = df[~df["label"].isin(["noise", "silence"])].copy()
    env_df = df[df["label"].isin(["noise", "silence"])].copy()
    
    # Unique speakers
    unique_speakers = sorted(speech_df["speaker_id"].unique())
    random.shuffle(unique_speakers)
    
    num_spk = len(unique_speakers)
    train_end = int(num_spk * 0.80)
    val_end = int(num_spk * 0.90)
    
    train_speakers = set(unique_speakers[:train_end])
    val_speakers = set(unique_speakers[train_end:val_end])
    test_speakers = set(unique_speakers[val_end:])
    
    print(f"Total Unique Speakers: {num_spk}")
    print(f"  Train Speakers: {len(train_speakers)}")
    print(f"  Val Speakers:   {len(val_speakers)}")
    print(f"  Test Speakers:  {len(test_speakers)}")
    
    # Assign speech samples based on speaker ID
    speech_df.loc[:, "split"] = speech_df["speaker_id"].apply(
        lambda spk: "train" if spk in train_speakers else ("val" if spk in val_speakers else "test")
    )
    
    # Shuffle environment noise/silence into train/val/test (80/10/10)
    env_indices = env_df.index.tolist()
    random.shuffle(env_indices)
    n_env = len(env_indices)
    n_tr = int(n_env * 0.80)
    n_va = int(n_env * 0.90)
    
    env_df.loc[env_indices[:n_tr], "split"] = "train"
    env_df.loc[env_indices[n_tr:n_va], "split"] = "val"
    env_df.loc[env_indices[n_va:], "split"] = "test"
    
    # Combine
    final_df = pd.concat([speech_df, env_df], ignore_index=True)
    
    # Verify zero speaker overlap
    tr_spk = set(final_df[final_df["split"] == "train"]["speaker_id"].unique()) - {"env_noise", "env_silence"}
    va_spk = set(final_df[final_df["split"] == "val"]["speaker_id"].unique()) - {"env_noise", "env_silence"}
    te_spk = set(final_df[final_df["split"] == "test"]["speaker_id"].unique()) - {"env_noise", "env_silence"}
    
    assert len(tr_spk.intersection(va_spk)) == 0, "Speaker leakage detected between Train and Val!"
    assert len(tr_spk.intersection(te_spk)) == 0, "Speaker leakage detected between Train and Test!"
    assert len(va_spk.intersection(te_spk)) == 0, "Speaker leakage detected between Val and Test!"
    
    print("\nSpeaker Leakage Verification PASSED (0 speaker overlap).")
    
    # Save train/val/test CSVs
    for split_name in ["train", "val", "test"]:
        sub_df = final_df[final_df["split"] == split_name]
        out_csv = os.path.join(DATASET_SPLITS, f"{split_name}.csv")
        sub_df.to_csv(out_csv, index=False)
        print(f"\nSaved {split_name}.csv ({len(sub_df)} samples):")
        print(sub_df["label"].value_counts())

if __name__ == "__main__":
    main()
