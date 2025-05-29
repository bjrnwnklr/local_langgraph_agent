# local_langgraph_agent (Python 3.12 • Windows 11 WSL2)

This repository bootstraps an **offline LangChain + LangGraph playground** that runs **entirely inside a WSL2 distro** on Windows 11. The agent uses **Python 3.12**, **Ollama**, and **Chroma**—so no cloud keys are required.

> **Goal** – go from an empty folder to a runnable agent you can chat with, plus an optional dev UI (LangGraph Studio).

---

## 0 · Prerequisites

| Tool   | Minimum version | Notes                                                                               |
| ------ | --------------- | ----------------------------------------------------------------------------------- |
| Python | **3.12.x**      | Use the Microsoft Store or [`pyenv`](https://github.com/pyenv/pyenv) _inside_ WSL2. |
| Git    | any             | version control                                                                     |
| Docker | latest          | only for `langgraph dev` hot‑reload server                                          |
| Ollama | 0.1.32 or newer | one‑line install inside WSL2 (see step 3)                                           |

> ℹ️ **Why Docker?** `langgraph dev` spins up its own container for live‑reload isolation. You can skip Docker entirely and just run `python main.py`.

---

## 1 · Project setup

```bash
# create & enter project folder (still inside WSL2)
mkdir local_langgraph_agent && cd local_langgraph_agent

# initialise Git (optional but recommended)
git init

# create Python 3.12 virtual environment
python3.12 -m venv .venv

# activate the venv (Linux-style path works in WSL)
source .venv/bin/activate
```

---

## 2 · Install dependencies

```bash
# upgrade pip
pip install -U pip

# install everything declared in requirements.txt
pip install -r requirements.txt
```

> 🔐 No API keys are needed; everything runs locally.

---

## 3 · Install and start Ollama

```bash
# one‑liner inside WSL2 Debian 12 (requires curl)
curl -fsSL https://ollama.com/install.sh | sh

# pull a 3–7 GB model (e.g. Llama 3)
ollama pull llama3

# start the daemon in the background (listens on 127.0.0.1:11434)
ollama serve &
```

_The port is bound to localhost inside WSL2; your Windows host can still reach it at **`http://localhost:11434`** thanks to the automatic port proxy._

---

## 4 · Project layout

```
local_langgraph_agent/
├─ main.py
├─ langgraph.json
├─ requirements.txt
└─ .env.example
```

### `langgraph.json`

```json
{
    "dependencies": ["./"],
    "graphs": {
        "research_agent": "./main.py:build_agent"
    },
    "python_version": "3.12"
}
```

> `langgraph dev` reads this file to locate your graph factory and to spin up the dev server with the correct interpreter.

---

## 5 · Run the agent

### Quick CLI test

```bash
python main.py
```

### Hot‑reload dev server + LangGraph Studio UI

```bash
langgraph dev
```

_Open _[**_http://localhost:2024_**](http://localhost:2024)_ in your Windows browser to inspect each node, time‑travel through state, and tweak code without restarting._

---

## 6 · Next steps

| Upgrade path           | How                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------- |
| **Add RAG**            | Ingest docs into Chroma and register a retriever tool.                                |
| **Multi‑agent graphs** | Replace `create_react_agent` with a hand‑rolled `StateGraph`.                         |
| **Package for prod**   | `langgraph build` emits a Docker image; you can run it on any host or publish to ACR. |

---

Happy hacking 🚀
