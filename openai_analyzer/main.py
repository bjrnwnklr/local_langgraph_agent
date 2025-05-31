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
import config  # loads environment variables (OpenAI API key, etc.)
from utils import csv_utils, openai_utils, clustering


def parse_arguments():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze HR chatbot interaction data using OpenAI API (classification, accuracy, clustering)."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input CSV file with chatbot interactions.",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="If set, limit analysis to the first 5 rows (for testing).",
    )
    parser.add_argument(
        "--pass",
        "-p",
        dest="pass_mode",
        choices=["first", "second", "both"],
        default="both",
        help="Which analysis pass to run: 'first' (classification only), 'second' (clustering only), or 'both' (default).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_path = Path(args.input)
    pass_mode = args.pass_mode
    is_test = args.test

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info(
        f"Starting analysis for file: {input_path} (mode={pass_mode}, test={is_test})"
    )

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
        logging.info(
            "Running second pass (embeddings, clustering, cluster labeling)..."
        )
        try:
            # Generate cluster assignments and labels
            cluster_labels = clustering.cluster_and_label(df)
            # Map cluster labels back to DataFrame (including noise as "Other")
            df["cluster"] = df[
                "cluster_id"
            ]  # (We'll set cluster_id in cluster_and_label)
            df["cluster_label"] = df["cluster_id"].map(cluster_labels)
            # Generate summary of top 10 clusters (if available)
            summary_df = clustering.summarize_clusters(df, top_n=10)
        except Exception as e:
            logging.error(f"Second pass analysis failed: {e}")
            return 1

    # Prepare output file paths (timestamped, based on input file name)
    # check if "output" folder exists
    outpath = input_path.parent / "output"
    if not outpath.exists():
        outpath.mkdir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = input_path.stem  # filename without extension
    enriched_path = outpath / f"{base_name}_enriched_{timestamp}.csv"
    summary_path = outpath / f"{base_name}_summary_{timestamp}.csv"

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
