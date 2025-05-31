"""
clustering.py - Clustering of question embeddings using DBSCAN and cluster labeling.
"""

import numpy as np
from sklearn.cluster import DBSCAN
import logging
from utils import openai_utils

# DBSCAN hyperparameters (can be tuned or made configurable)
EPSILON = 4  # maximum distance for points to be considered in the same cluster (for cosine distance)
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
        return pd.DataFrame(
            columns=["cluster_label", "count", "accuracy_pct", "avg_csat"]
        )

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
        csat_values = pd.to_numeric(
            grp["csat"], errors="coerce"
        )  # convert to numeric, non-numeric to NaN
        csat_values = csat_values.dropna()
        avg_csat = csat_values.mean() if not csat_values.empty else None

        summary_records.append(
            {
                "cluster_label": label,
                "count": count,
                "accuracy_pct": (
                    round(accuracy_pct, 2) if accuracy_pct is not None else "N/A"
                ),
                "avg_csat": round(avg_csat, 2) if avg_csat is not None else "N/A",
            }
        )
    # Sort records by count descending and take top N
    summary_records.sort(key=lambda x: x["count"], reverse=True)
    top_records = summary_records[:top_n]
    # Create DataFrame
    summary_df = pd.DataFrame(
        top_records, columns=["cluster_label", "count", "accuracy_pct", "avg_csat"]
    )
    return summary_df
