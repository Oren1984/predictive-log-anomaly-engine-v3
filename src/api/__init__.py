# src/api/__init__.py

# Purpose: Initialize the API package by importing the main application factory and settings class.

# Input: The __init__.py file imports the create_app function, 
# which is responsible for creating and configuring the Flask application instance, 
# and the Settings class, which encapsulates the configuration settings for the API.

# Output: By importing create_app and Settings in the __init__.py file,
# we make these components available for import when the api package is imported.

# Used by: The create_app function is used in the main entry point of 
# the application (e.g., run.py) to create the Flask app instance,
# and the Settings class is used throughout the application to access configuration settings.

"""Stage 7 — API package."""
from .app import create_app
from .settings import Settings

__all__ = ["create_app", "Settings"]
