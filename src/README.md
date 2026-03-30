# Source Code

This directory contains the core implementation of the predictive
log anomaly detection system.


## Main modules include:

- alerts          → alert generation and policies
- api             → FastAPI endpoints
- core            → shared contracts and core utilities
- data_layer      → data access and storage logic
- modeling        → anomaly detection models
- runtime         → real-time inference logic
- observability   → metrics and monitoring
- security        → authentication and API security

The system is designed as a modular architecture to support both
batch training and real-time anomaly detection.