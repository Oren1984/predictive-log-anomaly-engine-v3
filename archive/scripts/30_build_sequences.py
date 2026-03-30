# scripts/30_build_sequences.py

# Purpose: This script is responsible for building sequences of log events from the processed event data.
# It takes the prepared event data, organizes it into sequences based on session IDs or other relevant identifiers
# , and saves these sequences in a format suitable for training and testing the predictive log anomaly engine.

# Input: The script reads the processed event data (events_unified.csv) from the data/processed directory,
# which contains columns for timestamp, dataset, session_id, message, and label.
# It uses this data to create sequences of log events based on session IDs or other relevant identifiers.

# Output: The resulting sequences are saved in a format suitable for training and testing the predictive log anomaly engine,
# such as a CSV file or a serialized format (e.g., JSON, pickle) in the data/sequences directory.

# Used by: This script is typically run after preparing the events (using scripts/20_prepare_events.py)
# and before training the predictive log anomaly engine (using scripts/40_train_model.py).