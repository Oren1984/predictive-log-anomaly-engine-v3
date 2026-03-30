# scripts/90_run_api.py

# Purpose: This script is designed to run an API that serves the trained anomaly detection model.
# It will allow users to send log data to the API and receive predictions on whether the logs are anomalous or not.
# The API can be built using a web framework like Flask or FastAPI,
# and it will load the trained model from the models directory to make predictions.

# Input: Log data sent to the API in a specified format (e.g., JSON).

# Output: Predictions indicating whether the logs are anomalous or not.

# Used by: This script is used to deploy the trained anomaly detection model and make it accessible for real-time predictions.
# It is an essential part of the overall pipeline for anomaly detection in log data,
# as it allows users to interact with the model and utilize its capabilities in a practical setting.