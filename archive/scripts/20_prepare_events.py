# scripts/20_prepare_events.py

# Purpose: This script processes the raw log files from the HDFS and BGL datasets, extracts relevant information,
# and combines them into a unified CSV file.
# The resulting dataset will be used for building sequences in the next step of the predictive log anomaly engine pipeline.

# Input: The script reads raw log files (HDFS.log and BGL.log) and their corresponding label files (anomaly_label.csv for HDFS) from the data/raw directory.
# It processes the logs to extract session IDs, messages, and labels, and then combines them into a unified dataset.

# Output: The processed data is saved as a unified CSV file (events_unified.csv) in the data/processed directory,
# containing columns for timestamp, dataset, session_id, message, and label.

# Used by: This script is typically run after downloading the raw datasets (using scripts/10_download_data.py)
# and before building sequences (using scripts/30_build_sequences.py).

import re
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# -----------------------------
# HDFS PROCESSING
# -----------------------------
def process_hdfs():
    print("Processing HDFS...")

    # Search for log and label files in the raw data directory)
    log_candidates = list(RAW_PATH.rglob("HDFS.log"))
    if not log_candidates:
        raise FileNotFoundError("HDFS.log not found under data/raw")

    log_file = log_candidates[0]

    label_candidates = list(RAW_PATH.rglob("anomaly_label.csv"))
    if not label_candidates:
        raise FileNotFoundError("anomaly_label.csv not found under data/raw")

    label_file = label_candidates[0]

    print("Using log file:", log_file)
    print("Using label file:", label_file)

    labels = pd.read_csv(label_file)
    labels.columns = ["BlockId", "Label"]

    rows = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"(blk_-?\d+)", line)
            if match:
                block_id = match.group(1)
                rows.append({
                    "timestamp": None,
                    "dataset": "hdfs",
                    "session_id": block_id,
                    "message": line.strip()
                })

    df = pd.DataFrame(rows)

    df = df.merge(labels, left_on="session_id", right_on="BlockId", how="left")
    df["label"] = df["Label"].map({"Anomaly": 1, "Normal": 0})
    df["label"] = df["label"].fillna(0).astype(int)

    df.drop(columns=["BlockId", "Label"], inplace=True)

    return df


# -----------------------------
# BGL PROCESSING
# -----------------------------
def process_bgl():
    print("Processing BGL...")

    log_candidates = list(RAW_PATH.rglob("BGL.log"))
    if not log_candidates:
        raise FileNotFoundError("BGL.log not found under data/raw")

    log_file = log_candidates[0]

    print("Using BGL file:", log_file)

    rows = []
    window_size = 50
    window_id = 0

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i in range(0, len(lines), window_size):
        chunk = lines[i:i + window_size]

        for line in chunk:
            rows.append({
                "timestamp": None,
                "dataset": "bgl",
                "session_id": f"window_{window_id}",
                "message": line.strip(),
                "label": 1 if line.startswith("-") else 0
            })

        window_id += 1

    return pd.DataFrame(rows)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    hdfs_df = process_hdfs()
    bgl_df = process_bgl()

    final_df = pd.concat([hdfs_df, bgl_df], ignore_index=True)

    output_file = PROCESSED_PATH / "events_unified.csv"
    final_df.to_csv(output_file, index=False)

    print(f"\nUnified dataset saved to {output_file}")
    print("Total rows:", len(final_df))