Perfect, I’ll now implement the full HR analyzer solution using OpenAI’s latest Responses API, with a well-structured file and directory layout, prompt caching, and CLI support.

I'll get started and update you shortly with the completed code, organized into modules and ready for local use with your .env API key configuration.

# HR Chatbot Interaction Analysis Tool

This Python-based CLI tool reads an HR chatbot's Q\&A log from a CSV, classifies each query into HR categories, evaluates answer accuracy, and clusters similar queries for analysis. It produces an enriched CSV with new analysis columns and a summary CSV of top query clusters and metrics. The implementation emphasizes clarity, modularity, and cost-efficiency (using OpenAI’s latest API features and prompt caching).

## Features and Design Overview

-   **Command-Line Interface:** Accepts `--input` CSV path (required), `--test` flag (to analyze only first 5 rows), and `--pass` mode (`first`, `second`, or `both` – default is `both`).
-   **Input Validation:** Ensures the CSV has required columns (`user location`, `user query`, `answer`, `csat`), preventing execution on malformed data.
-   **First Pass (Classification & Accuracy):** Uses OpenAI’s Chat Completion API to classify each user query into one of 10 HR categories (e.g. Payroll, Onboarding, Benefits, etc.) and assess the chatbot’s answer accuracy as **fully correct**, **partially correct**, **incorrect**, or **not answered**. The prompt is structured with clear instructions and examples, following OpenAI best practices for formatting and role separation. The tool caches identical prompt content to avoid duplicate API calls and leverages OpenAI’s **prompt caching** feature to reduce cost on repeated prompt prefixes.
-   **Second Pass (Embedding & Clustering):** Generates embeddings for all queries using OpenAI’s efficient `text-embedding-3-small` model (1536-dimensional), which offers improved performance over Ada at a **5× lower cost**. It then uses DBSCAN clustering (density-based algorithm) to group semantically similar questions without requiring a predefined number of clusters. DBSCAN automatically treats outliers as noise rather than forcing them into clusters, ensuring only meaningful groupings are formed.
-   **Cluster Labeling:** For each cluster of questions, the tool uses GPT to generate a concise label summarizing the cluster’s topic. We feed a sample of questions from the cluster (up to 10) to the model with clear instructions to return only a short descriptive label.
-   **Output Files:** The tool outputs two CSV files, named after the input file with a timestamp:

    1. **Enriched CSV:** Contains all original columns plus new `classification`, `assessment`, `cluster`, and `cluster_label` columns for each query.
    2. **Summary CSV:** Lists the **top 10 clusters** (by frequency of questions) with their cluster label, question count, percentage of answers that were accurate (fully correct), and average CSAT score for that cluster (ignoring blank/NA CSAT entries).

Below we present the full implementation of the codebase. Each component is modular with appropriate documentation and logging for clarity and robustness. The project structure is as follows:

```
openai_analyzer/
├── main.py               # CLI entry point
├── config.py             # Environment config loader (API keys, settings)
├── prompts/
│   ├── first_pass_prompt.txt        # Prompt template for classification & accuracy
│   └── cluster_label_prompt.txt     # Prompt template for cluster labeling
├── utils/
│   ├── csv_utils.py      # CSV reading, validation, writing utilities
│   ├── openai_utils.py   # OpenAI API calls (classification, embedding, caching)
│   └── clustering.py     # DBSCAN clustering and cluster labeling logic
├── .env                  # API key and config values (not committed to repo)
└── requirements.txt      # Required Python packages
```

## main.py – Command-line Interface and Orchestration

The `main.py` script is the entry point. It parses command-line arguments, loads configuration (API keys, etc.), and orchestrates the two analysis passes. Based on the `--pass` argument, it invokes the first pass (classification & assessment), the second pass (clustering & labeling), or both in sequence. It uses functions from our utility modules and writes out the resulting CSV files. Logging is used to inform the user of progress.

```python
#!/usr/bin/env python3
"""
main.py - Entry point for the HR chatbot analysis tool.
Parses command-line arguments, orchestrates the analysis passes, and outputs results.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

# Local module imports
import config            # loads environment variables (OpenAI API key, etc.)
from utils import csv_utils, openai_utils, clustering

def parse_arguments():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze HR chatbot interaction data using OpenAI API (classification, accuracy, clustering)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file with chatbot interactions.")
    parser.add_argument("--test", "-t", action="store_true",
                        help="If set, limit analysis to the first 5 rows (for testing).")
    parser.add_argument("--pass", "-p", dest="pass_mode", choices=["first", "second", "both"], default="both",
                        help="Which analysis pass to run: 'first' (classification only), 'second' (clustering only), or 'both' (default).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_path = Path(args.input)
    pass_mode = args.pass_mode
    is_test = args.test

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Starting analysis for file: {input_path} (mode={pass_mode}, test={is_test})")

    # Load input CSV
    try:
        df = csv_utils.load_csv(input_path)
    except Exception as e:
        logging.error(f"Failed to load input CSV: {e}")
        return 1  # Exit with error code

    # If test flag is on, truncate to first 5 rows
    if is_test:
        df = df.head(5).copy()
        logging.info("Test mode enabled: only processing first 5 rows of data.")

    # Perform first pass: classification & accuracy assessment
    if pass_mode in ["first", "both"]:
        logging.info("Running first pass (classification and accuracy assessment)...")
        try:
            df = openai_utils.run_first_pass(df)
        except Exception as e:
            logging.error(f"First pass analysis failed: {e}")
            return 1

    # Perform second pass: embedding, clustering, cluster labeling
    summary_df = None
    if pass_mode in ["second", "both"]:
        logging.info("Running second pass (embeddings, clustering, cluster labeling)...")
        try:
            # Generate cluster assignments and labels
            cluster_labels = clustering.cluster_and_label(df)
            # Map cluster labels back to DataFrame (including noise as "Other")
            df["cluster"] = df["cluster_id"]  # (We'll set cluster_id in cluster_and_label)
            df["cluster_label"] = df["cluster_id"].map(cluster_labels)
            # Generate summary of top 10 clusters (if available)
            summary_df = clustering.summarize_clusters(df, top_n=10)
        except Exception as e:
            logging.error(f"Second pass analysis failed: {e}")
            return 1

    # Prepare output file paths (timestamped, based on input file name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = input_path.stem  # filename without extension
    enriched_path = input_path.parent / f"{base_name}_enriched_{timestamp}.csv"
    summary_path = input_path.parent / f"{base_name}_summary_{timestamp}.csv"

    # Save enriched CSV (with new columns)
    try:
        csv_utils.save_csv(df, enriched_path)
    except Exception as e:
        logging.error(f"Failed to write enriched CSV: {e}")
        return 1

    # Save summary CSV if second pass was done
    if summary_df is not None:
        try:
            csv_utils.save_csv(summary_df, summary_path)
        except Exception as e:
            logging.error(f"Failed to write summary CSV: {e}")
            return 1

    logging.info(f"Analysis complete. Enriched data saved to {enriched_path}")
    if summary_df is not None:
        logging.info(f"Cluster summary saved to {summary_path}")
    return 0

if __name__ == "__main__":
    main()
```

**Key points in `main.py`:**

-   We use Python’s `argparse` to handle CLI arguments. The `--pass` option is constrained to `'first'`, `'second'`, or `'both'` for safety.
-   Logging is configured at the INFO level to provide runtime feedback (file being processed, which pass is running, and any errors).
-   The input CSV is loaded via `csv_utils.load_csv()`, which validates required columns. If columns are missing or any error occurs, we log an error and exit gracefully.
-   Based on the selected pass mode, we call `openai_utils.run_first_pass()` to add classification and accuracy data, and `clustering.cluster_and_label()` (from `utils/clustering.py`) to perform DBSCAN clustering and get cluster labels.
-   We construct output filenames using the input name and a timestamp (format `YYYYMMDD_HHMMSS`) to avoid overwriting and to provide traceability. For example, input `chatlog.csv` might produce `chatlog_enriched_20250531_153010.csv` and `chatlog_summary_20250531_153010.csv`.
-   We use `csv_utils.save_csv()` to write the output CSVs. If only the first pass was run, `summary_df` remains `None` and no summary file is written. If second pass ran, we output the summary as well.
-   The script returns exit code 0 on success, or 1 on failure, following CLI best practices.

## config.py – Environment Configuration

The `config.py` module loads environment variables (from a local `.env` file) and sets global configuration used by the tool, such as the OpenAI API key and default model names. We use **python-dotenv** to load the `.env` file so that sensitive keys are not hard-coded. The OpenAI API key is required for API calls, and we ensure it is present before proceeding.

```python
"""
config.py - Loads environment variables and sets configuration for the analysis tool.
"""
import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Please add it to .env file or environment variables.")

# Set the OpenAI API key for the openai library
openai.api_key = OPENAI_API_KEY

# Default model configurations (can be adjusted via environment variables if needed)
OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4")  # latest model for classification tasks
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")  # efficient embedding model
OPENAI_LABEL_MODEL    = os.getenv("OPENAI_LABEL_MODEL", "gpt-3.5-turbo")  # model for cluster labeling (uses cheaper model by default)
```

**Explanation:** This module is imported at the start of `main.py`, so loading the environment and configuring the OpenAI library happens before any API calls. We define default models and allow overriding them via environment variables if needed (for example, switching to a smaller model to cut cost at some accuracy trade-off). The chosen defaults are:

-   **GPT-4** for classification/assessment (first pass) to maximize understanding and accuracy of analysis (developers can switch to GPT-3.5 via env if desired for cost savings).
-   **text-embedding-3-small** for embeddings, as it’s the latest small embedding model providing strong performance at low price.
-   **GPT-3.5-Turbo** for cluster labeling, as generating a short summary is a simpler task that GPT-3.5 can handle well at lower cost.

## Prompt Templates

We maintain prompt templates as text files in the `prompts/` directory. Keeping prompts in separate files makes them easier to review and tweak without altering code. The code will read these files and use them as the **system role content** for OpenAI API calls, ensuring consistent instructions across all calls (especially important for prompt caching benefits). We follow OpenAI’s prompt engineering best practices by clearly separating instructions from data and explicitly describing the desired output format.

### prompts/first_pass_prompt.txt

This prompt is used in the **first pass** for each Q\&A pair. It instructs the model to classify the user’s question into one of 10 HR categories and to assess the provided answer’s correctness. It also defines the meaning of each accuracy label for clarity. The model is asked to output the result in JSON format for easy parsing. By keeping these instructions constant and only swapping out the specific Q\&A content, we maximize reuse of the prompt prefix across calls – taking advantage of OpenAI’s prompt caching which **automatically discounts repeated prompt tokens**.

```text
You are an expert HR assistant analyzing employee HR questions and chatbot answers.

Your task is to output two things for each Q&A pair:
1. **Classification** – Identify which one of the following 10 HR categories best fits the user's question:
   - Payroll
   - Benefits
   - Leave/Time Off
   - Onboarding
   - Recruitment
   - Training/Development
   - Performance Management
   - Policy/Compliance
   - Compensation
   - Other (if none of the above applies)

2. **Accuracy Assessment** – Evaluate the chatbot's answer:
   - "fully correct" if the answer is completely correct and addresses the question.
   - "partially correct" if the answer is only partly correct or incomplete.
   - "incorrect" if the answer is wrong or irrelevant.
   - "not answered" if the answer does not actually attempt to answer the question (e.g., it deflects or says it cannot help).

**Format:** Provide the result as a JSON object with exactly two keys: "classification" and "assessment".

Respond ONLY with the JSON. Do not include any explanation.

```

For each Q\&A in the CSV, the code will append the specific content to this prompt. The actual **user message** sent to the model will look like:

```
User Question: <user query text>
Chatbot Answer: <chatbot answer text>
```

This user message is appended after the above instructions (which are given as the system message in the API call). The model’s task is to output JSON, for example:

```json
{ "classification": "Payroll", "assessment": "fully correct" }
```

By providing a consistent instruction prefix and output format, we ensure reliable, easily parseable responses and minimize token usage per call. (The OpenAI API will cache the instruction prefix once it exceeds 1024 tokens across requests, reducing cost for subsequent calls.)

### prompts/cluster_label_prompt.txt

This prompt is used in the **second pass** to label clusters of similar questions. After clustering, for each cluster we feed a sample of the questions to the model and ask for a concise summary label. We instruct the model to be succinct and only return the label text.

```text
You are an AI assistant specialized in summarizing topics.

Given a list of user questions that all relate to the same topic, generate a short descriptive label for that topic.
- The label should be a few words (3-5 words) capturing the essence of the questions.
- Do NOT return a full sentence or any explanations, only the brief label.

```

When calling the API for a specific cluster, the code will format a user message listing a subset of the cluster’s questions, for example:

```
Questions in cluster:
1. How do I update my direct deposit for payroll?
2. When are paychecks issued if payday falls on a holiday?
3. What is the process to correct a payroll error?
...
```

The model (using this prompt as system instructions) will then output something like:

```
Payroll and Paycheck Issues
```

This becomes the `cluster_label` for that cluster. By restricting the output to just a few words, we ensure the label is suitable as a concise descriptor in the summary.

## utils/csv_utils.py – CSV Loading and Saving Utilities

This module contains helper functions to read the input CSV and write output CSVs, including validating the presence of required columns. We use **pandas** for convenient CSV handling. The `load_csv()` function checks that all required columns are present, and if not, raises an error. The `save_csv()` function wraps `DataFrame.to_csv` and could include additional logic if needed (for example, ensuring certain encodings or handling index).

```python
"""
csv_utils.py - Functions for reading from and writing to CSV files, with validation.
"""
import pandas as pd

REQUIRED_COLUMNS = ["user location", "user query", "answer", "csat"]

def load_csv(file_path):
    """
    Read the CSV file into a pandas DataFrame and validate required columns.
    :param file_path: Path to the CSV file.
    :return: pandas DataFrame with the CSV contents.
    :raises: ValueError if required columns are missing.
    """
    # Read CSV (assume default encoding and comma delimiter; adjust if needed)
    df = pd.read_csv(file_path)
    # Validate that all required columns are present (case-sensitive match)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")
    return df

def save_csv(df, file_path):
    """
    Save the DataFrame to a CSV file.
    :param df: pandas DataFrame to save.
    :param file_path: Destination file path.
    """
    df.to_csv(file_path, index=False)
```

**Notes:**

-   We treat column names as case-sensitive and exact matches. If the input file has different casing or slight naming differences, the user should adjust it or we could enhance the code to be case-insensitive/flexible. For now, it expects exactly "user location", "user query", "answer", "csat".
-   The `REQUIRED_COLUMNS` list is defined at the top for easy maintenance (if tomorrow the schema changes, we update it in one place).
-   The `load_csv` uses `pd.read_csv` without additional parameters, assuming a standard CSV. If the input were large or needed specific parsing (separators, encoding), we could adjust or expose those as config.
-   We explicitly do not write the DataFrame index to CSV (`index=False`) since these are record-oriented outputs.

## utils/openai_utils.py – OpenAI API Calls and Caching

This module encapsulates all interactions with the OpenAI API: classification/assessment calls, embedding generation, and cluster labeling calls. It also implements caching to avoid redundant API calls for identical inputs, reducing cost and latency.

Key components:

-   **`run_first_pass(df)`** – Iterates over the DataFrame rows and obtains classification & assessment for each Q\&A pair using `ChatCompletion`. It adds two new columns to the DataFrame: `classification` and `assessment`. Caching is used to skip API calls if a query-answer pair was seen before.
-   **`classify_query(question, answer)`** – Internal helper that prepares the prompt and calls the OpenAI ChatCompletion API to get a classification/assessment for one Q\&A pair. It uses the prompt from `first_pass_prompt.txt` as the system message, and formats the Q\&A as the user message. The model’s JSON response is parsed and returned.
-   **`get_embeddings(text_list)`** – Calls OpenAI’s embedding API (`openai.Embedding.create`) on a batch of input texts (user queries) to get their embedding vectors. We use batching for efficiency. The result is a list of vectors (one per input text).
-   **`get_cluster_label(questions_list)`** – Calls the OpenAI ChatCompletion API with the cluster labeling prompt to generate a label for a given cluster of questions.

We utilize the environment settings from `config.py` for model names and API keys. Logging is added to record each API call and any cache hits. Basic error handling and retry logic are included to make the pipeline robust to transient API issues.

```python
"""
openai_utils.py - OpenAI API integration (classification, embedding, and clustering label generation), with caching.
"""
import json
import logging
import openai
import time
from config import OPENAI_COMPLETION_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_LABEL_MODEL

# Simple in-memory cache for classification results to avoid duplicate API calls
_classification_cache = {}

def classify_query(user_question, chatbot_answer):
    """
    Use OpenAI ChatCompletion to classify an HR question and assess answer accuracy.
    Returns a tuple: (classification, assessment).
    """
    # Prepare cache key
    cache_key = (user_question.strip(), chatbot_answer.strip())
    if cache_key in _classification_cache:
        logging.debug("Cache hit for classification of question: '%s'" % user_question)
        return _classification_cache[cache_key]

    # Read the system prompt instructions from file
    with open("prompts/first_pass_prompt.txt", "r") as f:
        system_prompt = f.read().strip()

    # Construct the messages for ChatCompletion
    user_message = f"User Question: {user_question}\nChatbot Answer: {chatbot_answer}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Call the OpenAI ChatCompletion API
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_COMPLETION_MODEL,
            messages=messages,
            temperature=0  # using deterministic output for consistency
        )
    except Exception as e:
        logging.error(f"OpenAI API error during classification: {e}")
        # Simple retry once after a brief pause
        time.sleep(2)
        response = openai.ChatCompletion.create(
            model=OPENAI_COMPLETION_MODEL,
            messages=messages,
            temperature=0
        )
    # Extract the content of the assistant's reply
    reply_content = response["choices"][0]["message"]["content"].strip()
    # Parse the JSON output
    classification = None
    assessment = None
    try:
        result_json = json.loads(reply_content)
        classification = result_json.get("classification")
        assessment   = result_json.get("assessment")
    except json.JSONDecodeError:
        # If the model didn't respond with valid JSON (should not happen given our prompt),
        # we attempt to fix minor format issues or log an error.
        logging.warning("Failed to parse JSON from model, attempting to fix format.")
        # Basic fix: e.g., remove trailing comma or surrounding text if any.
        try:
            import re
            json_str = re.search(r'\{.*\}', reply_content).group(0)
            result_json = json.loads(json_str)
            classification = result_json.get("classification")
            assessment   = result_json.get("assessment")
        except Exception:
            logging.error(f"Could not parse classification JSON: {reply_content}")
            # Default to "Other" and "not answered" if parsing fails
            classification = classification or "Other"
            assessment = assessment or "not answered"

    # Cache the result and return
    _classification_cache[cache_key] = (classification, assessment)
    return classification, assessment

def run_first_pass(df):
    """
    Process the DataFrame by adding 'classification' and 'assessment' for each row using OpenAI.
    """
    classifications = []
    assessments = []
    for idx, row in df.iterrows():
        question = str(row["user query"])
        answer   = str(row["answer"])
        classification, assessment = classify_query(question, answer)
        classifications.append(classification)
        assessments.append(assessment)
        # Optionally log progress for large files
        if (idx + 1) % 50 == 0:
            logging.info(f"Processed {idx+1} rows for classification")
    # Add new columns to the DataFrame
    df["classification"] = classifications
    df["assessment"] = assessments
    return df

def get_embeddings(texts):
    """
    Fetch embeddings for a list of texts (user queries) using the OpenAI Embedding API.
    :param texts: List of strings.
    :return: List of embedding vectors (list of floats).
    """
    # OpenAI Embedding API allows batch processing. We handle in batches if list is large.
    embeddings = []
    batch_size = 1000  # batch size can be tuned
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = openai.Embedding.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        except Exception as e:
            logging.error(f"Embedding API call failed: {e}")
            raise
        # Extract embeddings from response (assuming 'data' list of {'embedding': [...]})
        for item in response["data"]:
            embeddings.append(item["embedding"])
    return embeddings

def get_cluster_label(question_list):
    """
    Generate a concise label for a cluster of questions using OpenAI.
    :param question_list: List of representative questions from the cluster.
    :return: A short descriptive label (string).
    """
    # Read cluster labeling prompt
    with open("prompts/cluster_label_prompt.txt", "r") as f:
        system_prompt = f.read().strip()

    # Format the questions into a single string for the user message
    questions_text = ""
    for i, q in enumerate(question_list, start=1):
        questions_text += f"{i}. {q}\n"
    user_message = f"Questions in cluster:\n{questions_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_LABEL_MODEL,
            messages=messages,
            temperature=0
        )
    except Exception as e:
        logging.error(f"OpenAI API error during cluster labeling: {e}")
        # No retry here, just propagate error or return a default label
        raise
    label = response["choices"][0]["message"]["content"].strip()
    # Post-process label: ensure it's a single line and not too long
    label = label.split("\n")[0]  # take first line if model returned multiple lines
    return label
```

**Highlights of `openai_utils.py`:**

-   We set `temperature=0` for classification and labeling calls to ensure deterministic outputs; this is important for consistent results (the task is analytical, not creative).
-   **Caching:** `_classification_cache` is a simple dictionary mapping `(question, answer)` to the model’s result. This handles scenarios where the same user query and answer appear multiple times (avoiding repeated API charges). While not extremely common in a single CSV of conversations, it can happen (e.g., repeated FAQs). Additionally, the reuse of prompt prefix across different calls is handled by OpenAI’s internal caching mechanism automatically. Our cache complements this by also skipping the API call entirely for exact repeats.
-   We read prompt files from disk for each call. This ensures if you tweak the prompt text, the code uses the latest version on each invocation. The slight overhead of reading from file is negligible compared to the API call latency. (Alternatively, one could read the prompt once and keep it in a module-level variable to avoid repeated disk I/O.)
-   **Error handling:** If the API call fails (due to network issues or rate limits), we catch exceptions and retry once after a short delay for classification. For embeddings, we propagate the exception (which will be caught in the calling function in `main.py`). For cluster labeling, we simply propagate (assuming cluster labeling is not critical enough to retry, or could be retried similarly if needed).
-   **JSON parsing:** We expect the model to return well-formed JSON. In case of a formatting surprise, we attempt to extract a JSON substring via regex as a fallback. This is rarely needed because our prompt is designed to elicit proper JSON. Logging will warn us if a response wasn’t directly parsed, and we default to `"Other"/"not answered"` if completely unparseable. This conservative approach ensures the pipeline doesn’t crash on a single malformed response.
-   **Batch embeddings:** We send up to 1000 queries in one batch to `openai.Embedding.create` (the API supports large batch sizes; this number can be tuned based on memory constraints or API limits). This is much faster and cheaper than calling one by one. All embedding vectors returned are stored in order corresponding to the input list.
-   The `get_cluster_label` function uses the cluster’s questions (up to 10) to ask the model for a label. We join questions with numbering to provide clear separation. We instruct the model not to produce explanations, just a brief phrase. We strip the result and take the first line to guard against any extraneous output.

## utils/clustering.py – Clustering and Cluster Analysis

This module handles the second pass: using embeddings to cluster similar questions, and analyzing those clusters. It uses **scikit-learn’s** DBSCAN algorithm for clustering and relies on `openai_utils.get_cluster_label` for labeling. It also provides a summary of clusters with required metrics.

Key functions:

-   **`cluster_and_label(df)`** – Computes embeddings for all questions, performs DBSCAN clustering, and returns a dictionary of cluster labels (mapping cluster id -> cluster label). It also attaches the cluster assignments to the DataFrame (`df["cluster_id"]`).
-   **`summarize_clusters(df, top_n=10)`** – Generates a summary DataFrame for the top N clusters by size. It calculates each cluster’s question count, percentage of fully correct answers, and average CSAT.

We take care to exclude noise points (DBSCAN label `-1`) from labeling and summary metrics, since those are disparate questions that didn’t form a meaningful cluster.

```python
"""
clustering.py - Clustering of question embeddings using DBSCAN and cluster labeling.
"""
import numpy as np
from sklearn.cluster import DBSCAN
import logging
from utils import openai_utils

# DBSCAN hyperparameters (can be tuned or made configurable)
EPSILON = 0.15  # maximum distance for points to be considered in the same cluster (for cosine distance)
MIN_SAMPLES = 3  # minimum points to form a dense cluster

def cluster_and_label(df):
    """
    Perform DBSCAN clustering on the user query embeddings and label each cluster.
    Adds a 'cluster_id' column to the DataFrame.
    Returns a dict mapping cluster_id -> cluster_label.
    """
    # Get embeddings for all user queries
    queries = df["user query"].astype(str).tolist()
    logging.info("Generating embeddings for %d queries..." % len(queries))
    embeddings = openai_utils.get_embeddings(queries)
    # Convert to NumPy array for clustering
    X = np.array(embeddings, dtype=float)
    # Normalize embeddings to unit length (for cosine similarity-based clustering)
    # (Cosine distance = 1 - cosine_similarity; by normalizing, Euclidean distance correlates with cosine distance)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Perform DBSCAN clustering with cosine distance metric
    clustering_model = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric="cosine")
    labels = clustering_model.fit_predict(X_norm)
    df["cluster_id"] = labels  # add cluster assignments to DataFrame

    # Determine unique clusters (excluding noise label -1)
    unique_clusters = [c for c in set(labels) if c != -1]
    cluster_labels = {}  # map from cluster id to descriptive label
    for cluster_id in unique_clusters:
        # Get all questions in this cluster
        cluster_questions = df[df["cluster_id"] == cluster_id]["user query"].tolist()
        # If cluster has more than 10 questions, sample 10 for labeling to keep prompt size reasonable
        sample_questions = cluster_questions[:10]
        try:
            label = openai_utils.get_cluster_label(sample_questions)
        except Exception as e:
            logging.error(f"Failed to get label for cluster {cluster_id}: {e}")
            label = "Unknown"
        cluster_labels[cluster_id] = label

    # Optionally label noise/outliers as "Other" for completeness
    if -1 in set(labels):
        cluster_labels[-1] = "Other"  # all noise points considered "Other"
    return cluster_labels

def summarize_clusters(df, top_n=10):
    """
    Create a summary of the top N clusters by frequency.
    Calculates count, % fully correct answers, and average CSAT for each cluster.
    Excludes the noise cluster (-1) from consideration.
    :return: pandas DataFrame with summary information.
    """
    import pandas as pd
    # Filter out noise points (cluster -1) for summary
    clustered_df = df[df["cluster_id"] != -1]
    if clustered_df.empty:
        logging.warning("No clusters (other than noise) found to summarize.")
        return pd.DataFrame(columns=["cluster_label", "count", "accuracy_pct", "avg_csat"])

    # Group by cluster_id
    group = clustered_df.groupby("cluster_id")
    summary_records = []
    for cluster_id, grp in group:
        label = grp["cluster_label"].iloc[0]  # all in group have same label
        count = len(grp)
        # Calculate percentage of fully correct answers in this cluster
        if "assessment" in grp.columns:
            correct_count = sum(grp["assessment"] == "fully correct")
            accuracy_pct = (correct_count / count) * 100
        else:
            accuracy_pct = None  # if no assessment data available
        # Calculate average CSAT for this cluster (exclude blanks or NaN)
        csat_values = pd.to_numeric(grp["csat"], errors='coerce')  # convert to numeric, non-numeric to NaN
        csat_values = csat_values.dropna()
        avg_csat = csat_values.mean() if not csat_values.empty else None

        summary_records.append({
            "cluster_label": label,
            "count": count,
            "accuracy_pct": round(accuracy_pct, 2) if accuracy_pct is not None else "N/A",
            "avg_csat": round(avg_csat, 2) if avg_csat is not None else "N/A"
        })
    # Sort records by count descending and take top N
    summary_records.sort(key=lambda x: x["count"], reverse=True)
    top_records = summary_records[:top_n]
    # Create DataFrame
    summary_df = pd.DataFrame(top_records, columns=["cluster_label", "count", "accuracy_pct", "avg_csat"])
    return summary_df
```

**Explanation:**

-   We use **cosine distance** for DBSCAN (`metric="cosine"`) because it’s well-suited for high-dimensional text embeddings (we care about the angle between vectors, not raw Euclidean distance). We normalize all embedding vectors to unit length so that Euclidean distance is directly related to cosine similarity.
-   `EPSILON = 0.15` and `MIN_SAMPLES = 3` are chosen as defaults for clustering. These can be tuned: a smaller epsilon makes clusters tighter (higher similarity), and min_samples=3 ensures we only consider a cluster valid if it has at least 3 points. With these settings, queries need to be very similar to end up in a cluster, which suits our use case of finding frequently asked question themes.
-   After clustering, we get a `labels` array indicating cluster membership for each query (with `-1` for noise/outlier points that didn’t fit any cluster). We add this as `cluster_id` in the DataFrame for reference.
-   We then generate a label for each cluster (except noise) by taking up to 10 questions from that cluster. We catch exceptions from the OpenAI API and label those clusters as "Unknown" in case of failure, to ensure the pipeline continues. We add a label for `-1` as "Other" to mark outliers in the enriched DataFrame (though we will exclude them from the top-10 summary).
-   The `summarize_clusters` function filters out cluster `-1` and groups the DataFrame by cluster_id. For each cluster, it retrieves the label (all entries in the cluster share the same `cluster_label` from earlier assignment), the count of questions, the percentage of **fully correct** answers (if the assessment column is present), and the average CSAT score. We convert CSAT to numeric; non-numeric or blank entries are ignored in the average. If no CSAT is available in a cluster, we mark it as "N/A". Similarly, if for some reason assessment wasn’t run (like if user chose `--pass second` on a file with no existing assessment data), we mark accuracy as "N/A".
-   We round the accuracy percentage and CSAT to two decimals for neatness. The output summary DataFrame has columns: `cluster_label`, `count`, `accuracy_pct`, `avg_csat`. We sort clusters by size and take the top 10. If there are fewer than 10 clusters, it will naturally list all available. If no clusters (other than noise) were found (e.g., if every point was noise because epsilon was too low or data is very diverse), we return an empty DataFrame with the expected columns and log a warning.

## .env – Environment Variables

An example `.env` file is shown below. It should reside in the project root (`openai_analyzer/.env`) and contain your OpenAI API key and any overrides for model choices if desired:

```text
OPENAI_API_KEY=<your_openai_api_key_here>
# Uncomment and adjust the following to override default models (optional):
# OPENAI_COMPLETION_MODEL=gpt-3.5-turbo
# OPENAI_EMBEDDING_MODEL=text-embedding-3-large
# OPENAI_LABEL_MODEL=gpt-4
```

The `.env` file is loaded by `config.py`. We recommend keeping this file secure and not checking it into version control, as it contains your secret API key.

## requirements.txt – Python Dependencies

The project requires Python 3.12 and the following libraries (versions can be adjusted as needed):

```text
openai
pandas
scikit-learn
python-dotenv
```

> **Note:** `argparse` and `logging` are part of Python’s standard library and do not need to be installed via pip. They are included here for completeness because we use them in the code.

Make sure to install these packages (e.g., via `pip install -r requirements.txt`) before running the tool.

## Running the Tool

To use the tool, run `main.py` with the required arguments. For example:

```bash
$ python main.py --input hr_chatbot_logs.csv --pass both
```

This will process the file `hr_chatbot_logs.csv` through both analysis passes. If you want to test on a small subset first, use the `--test` flag:

```bash
$ python main.py --input hr_chatbot_logs.csv --test --pass first
```

This command would only classify and assess the first 5 rows of the input file (quick test of first pass).

After running, you will find:

-   An enriched CSV (e.g., `hr_chatbot_logs_enriched_20250531_153010.csv`) with new columns: **classification**, **assessment**, **cluster**, **cluster_label** for each query.
-   A summary CSV (e.g., `hr_chatbot_logs_summary_20250531_153010.csv`) listing up to 10 largest clusters with their label, question count, answer accuracy percentage, and average CSAT score.

These outputs enable HR teams to identify common inquiry themes and the chatbot’s performance on each, helping to pinpoint areas for improvement (e.g., topics with low accuracy or satisfaction).
