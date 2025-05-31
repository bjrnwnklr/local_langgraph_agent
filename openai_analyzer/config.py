"""
config.py - Loads environment variables and sets configuration for the analysis tool.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# load API keys
# 1. Compute project root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".local.env")

# Retrieve OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please add it to .env file or environment variables."
    )

# Set the OpenAI API key for the openai library
# openai.api_key = OPENAI_API_KEY

# Default model configurations (can be adjusted via environment variables if needed)
OPENAI_COMPLETION_MODEL = os.getenv(
    "OPENAI_COMPLETION_MODEL", "gpt-4.1-mini"
)  # latest model for classification tasks
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)  # efficient embedding model
OPENAI_LABEL_MODEL = os.getenv(
    "OPENAI_LABEL_MODEL", "gpt-4.1-mini"
)  # model for cluster labeling (uses cheaper model by default)

# create OpenAI instance
client = OpenAI()
