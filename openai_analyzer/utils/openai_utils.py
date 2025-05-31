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
        {"role": "user", "content": user_message},
    ]

    # Call the OpenAI ChatCompletion API
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_COMPLETION_MODEL,
            messages=messages,
            temperature=0,  # using deterministic output for consistency
        )
    except Exception as e:
        logging.error(f"OpenAI API error during classification: {e}")
        # Simple retry once after a brief pause
        time.sleep(2)
        response = openai.ChatCompletion.create(
            model=OPENAI_COMPLETION_MODEL, messages=messages, temperature=0
        )
    # Extract the content of the assistant's reply
    reply_content = response["choices"][0]["message"]["content"].strip()
    # Parse the JSON output
    classification = None
    assessment = None
    try:
        result_json = json.loads(reply_content)
        classification = result_json.get("classification")
        assessment = result_json.get("assessment")
    except json.JSONDecodeError:
        # If the model didn't respond with valid JSON (should not happen given our prompt),
        # we attempt to fix minor format issues or log an error.
        logging.warning("Failed to parse JSON from model, attempting to fix format.")
        # Basic fix: e.g., remove trailing comma or surrounding text if any.
        try:
            import re

            json_str = re.search(r"\{.*\}", reply_content).group(0)
            result_json = json.loads(json_str)
            classification = result_json.get("classification")
            assessment = result_json.get("assessment")
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
        answer = str(row["answer"])
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
        batch = texts[i : i + batch_size]
        try:
            response = openai.Embedding.create(
                model=OPENAI_EMBEDDING_MODEL, input=batch
            )
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
        {"role": "user", "content": user_message},
    ]
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_LABEL_MODEL, messages=messages, temperature=0
        )
    except Exception as e:
        logging.error(f"OpenAI API error during cluster labeling: {e}")
        # No retry here, just propagate error or return a default label
        raise
    label = response["choices"][0]["message"]["content"].strip()
    # Post-process label: ensure it's a single line and not too long
    label = label.split("\n")[0]  # take first line if model returned multiple lines
    return label
