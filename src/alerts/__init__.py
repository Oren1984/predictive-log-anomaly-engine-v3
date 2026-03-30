# src/alerts/__init__.py

# Purpose: Define the public API for the alerts package, which includes Alert, 
# AlertPolicy, AlertManager, and N8nWebhookClient.

# Input: This file imports the necessary classes from their respective modules 
# and defines the __all__ variable to specify what is exported when using 'from alerts import *'. 
# It serves as a central point for importing these classes in other parts 
# of the codebase, such as tests and scripts.

# Output: When this module is imported, it provides access to the Alert, 
# AlertPolicy, AlertManager, and N8nWebhookClient classes. 
# This allows other modules to import these classes directly from the alerts 
# package without needing to know their specific locations within the package.

# Used by: This module is used by various test files (e.g., test_stage_06_n8n_outbox.py, test_stage_06_dedup_cooldown.py) 
# and the main API implementation in src.api.app.py to create and manage alerts, 
# define alert policies, and send alerts to n8n via webhooks. 
# It is a core part of the alerting functionality in the demo pipeline.

"""Stage 6 — Alerts package: Alert, AlertPolicy, AlertManager, N8nWebhookClient."""
from .manager import AlertManager
from .models import Alert, AlertPolicy
from .n8n_client import N8nWebhookClient

__all__ = [
    "Alert",
    "AlertPolicy",
    "AlertManager",
    "N8nWebhookClient",
]
