import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
import numpy as np

# load API keys
# 1. Compute project root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".local.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .local.env")

# create OpenAI instance
client = OpenAI()

# HR categories
HR_CATEGORIES = [
    "Recruitment & Onboarding",
    "Payroll & Compensation",
    "Benefits & Insurance",
    "Leave & Attendance",
    "Training & Development",
    "Performance & Promotion",
    "Policies & Compliance",
    "Employee Relations & Grievances",
    "Travel & Expense",
    "Offboarding & Retirement",
]

# Prompts
FIRST_PASS_TEMPLATE = """
You are an expert HR analyst. Classify each user question into one of the following categories: {categories}.
Evaluate each provided answer for accuracy as: fully correct, partially correct, incorrect, or not answered.
Respond as a JSON list of objects with keys: id, classification, assessment.

Questions and Answers:
{items}
"""

CLUSTER_LABEL_TEMPLATE = """
You are an HR expert. Provide a concise descriptive label for the following cluster of similar HR questions:

Questions:
{questions}

Cluster Label:
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze HR chatbot interactions.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument(
        "--test", "-t", action="store_true", help="Run in test mode (first 5 rows only)"
    )
    parser.add_argument(
        "--pass",
        dest="stage",
        choices=["first", "second", "both"],
        default="both",
        help="Which analysis pass to run",
    )
    return parser.parse_args()


def validate_csv(path):
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    expected = {"user location", "user query", "answer", "csat"}
    if not expected.issubset(df.columns):
        logging.error(f"CSV missing required columns. Found: {df.columns.tolist()}")
        sys.exit(1)
    return df


def batch_first_pass(df):
    results = []
    batch_size = 10
    items = []
    for idx, row in df.iterrows():
        items.append(
            {
                "id": int(idx),
                "location": row["user location"],
                "question": row["user query"],
                "answer": row["answer"],
            }
        )
        # process batch
        if len(items) == batch_size or idx == df.index[-1]:
            prompt_items = "\n".join(
                [
                    f"{item['id']}. {{\"id\":{item['id']}, \"location\":\"{item['location']}\", \"question\":\"{item['question']}\", \"answer\":\"{item['answer']}\"}}"
                    for item in items
                ]
            )
            prompt = FIRST_PASS_TEMPLATE.format(
                categories=", ".join(HR_CATEGORIES), items=prompt_items
            )
            logging.info(f"Calling OpenAI for batch starting at id {items[0]['id']}")
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
            )
            content = resp.choices[0].message.content
            batch_res = pd.read_json(content)
            results.append(batch_res)
            items = []
    combined = pd.concat(results).set_index("id").sort_index()
    df["classification"] = combined["classification"]
    df["assessment"] = combined["assessment"]
    return df


def second_pass_with_embeddings(df):
    # Generate embeddings
    questions = df["user query"].tolist()
    logging.info("Requesting embeddings for questions...")
    embed_resp = client.embeddings.create(
        model="text-embedding-3-small", input=questions
    )
    print(embed_resp)
    vectors = np.array([e["embedding"] for e in embed_resp])
    # Cluster
    logging.info("Clustering embeddings...")
    dbscan = DBSCAN(eps=0.4, min_samples=2, metric="cosine")
    labels = dbscan.fit_predict(vectors)
    df["cluster"] = labels
    # Label clusters
    cluster_labels = {}
    for cluster in set(labels):
        if cluster < 0:
            continue  # skip noise
        sample_qs = df[df["cluster"] == cluster]["user query"].head(5).tolist()
        prompt = CLUSTER_LABEL_TEMPLATE.format(
            questions="\n".join([f"- {q}" for q in sample_qs])
        )
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
        )
        label = resp.choices[0].message.content.strip().strip('"')
        cluster_labels[cluster] = label
    df["cluster_label"] = df["cluster"].map(cluster_labels).fillna("Other")
    # Aggregate summary
    summary = (
        df[df["cluster"] >= 0]
        .groupby("cluster")
        .agg(
            question_count=("user query", "count"),
            accuracy_percent=(
                "assessment",
                lambda x: (x == "fully correct").mean() * 100,
            ),
            average_csat=(
                "csat",
                lambda x: pd.to_numeric(x, errors="coerce").dropna().mean(),
            ),
        )
        .reset_index()
    )
    summary["cluster_label"] = summary["cluster"].map(cluster_labels)
    summary = summary.sort_values("question_count", ascending=False).head(10)
    return df, summary


def main():
    args = parse_args()
    df = validate_csv(args.input)
    if args.test:
        df = df.head(5)
    if args.stage in ("first", "both"):
        df = batch_first_pass(df)
    # Save intermediate if only first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(args.input)
    analysis_file = f"{base}_analysis_{timestamp}{ext}"
    df.to_csv(analysis_file, index=False)
    logging.info(f"First-pass analysis saved to {analysis_file}")

    if args.stage in ("second", "both"):
        df2, summary = second_pass_with_embeddings(df)
        summary_file = f"{base}_summary_{timestamp}.csv"
        summary.to_csv(summary_file, index=False)
        logging.info(f"Second-pass summary saved to {summary_file}")


if __name__ == "__main__":
    main()
