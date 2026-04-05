# Source Code

This directory contains the core implementation of the predictive
log anomaly detection system.


## Main modules include:

- alerts          → alert generation and policies
- api             → FastAPI endpoints
- health          → health checks and readiness composition
- data_layer      → data access and storage logic
- data            → legacy/secondary data model helpers (limited usage)
- modeling        → anomaly detection models
- runtime         → real-time inference logic
- observability   → metrics and monitoring
- security        → authentication and API security
- semantic        → optional V3 semantic explanation/similarity layer

The system is designed as a modular architecture to support both
batch training and real-time anomaly detection.