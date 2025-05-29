#!/usr/bin/env python3
"""
Local LangGraph starter.

✓ Runs fully offline (Ollama + Chroma)
✓ Can be inspected live via `langgraph dev`
"""

import os
from pathlib import Path

from langgraph.prebuilt import create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import (
    ChatOllama,
)  # local LLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = Path("./data/chroma")  # persistent vector store
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
TEMPERATURE = float(os.getenv("TEMP", 0.7))


def make_llm():
    """Connect to the running Ollama daemon."""
    return ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)


def make_retriever():
    """Create (or reuse) a Chroma collection for long-term memory."""
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectordb = Chroma(
        persist_directory=str(DATA_PATH),
        embedding_function=embeddings,
        collection_name="agent_knowledge",
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})


def build_agent():
    llm = make_llm()
    memory = ConversationBufferMemory(return_messages=True)
    agent = create_react_agent(
        model=llm,
        tools=[],  # add tools later (e.g. browser, file-ops)
        prompt="You are a helpful research assistant.",
        memory=memory,
    )
    return agent


if __name__ == "__main__":
    agent = build_agent()
    print("💡  Type 'quit' to exit.")
    while True:
        user = input("\n👤  ")
        if user.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": user}]})
        print("🤖 ", result.content)
