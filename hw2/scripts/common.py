"""Общий код hw2: клиент LM Studio, логирование, пути"""
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = DATA / "docs"
GOLDEN = DATA / "golden"
INDEX_DIR = ROOT / "index"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

BASE_URL = "http://127.0.0.1:1234/v1"
CHAT_MODEL = "qwen/qwen3.5-9b"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

EMBED_DIM = 768
CHUNK_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50
TOP_K_DEFAULT = 5


def chat(messages, tools=None, max_tokens=512, log=None):
    """Один запрос к чат-модели, возвращает (content, tool_calls, usage)"""
    body = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    usage = dict(data.get("usage", {}))
    # Qwen с thinking может потратить весь бюджет на reasoning_conten;
    # при finish_reason=length ответ обрезан - повторяем с удвоенным лимитом
    # Токены обоих вызовов суммируются, чтобы стоимость не занижалась
    if (data["choices"][0].get("finish_reason") == "length"
            and not msg.get("tool_calls")):
        body["max_tokens"] = max_tokens * 2
        r = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        retry_usage = data.get("usage", {})
        usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0) + retry_usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0) + retry_usage.get("completion_tokens", 0),
        }
    if log:
        log.write(json.dumps({
            "ts": time.time() - t0,
            "event": "chat",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": data["choices"][0].get("finish_reason"),
        }) + "\n")
    tool_calls = [
        {"id": tc["id"], "name": tc["function"]["name"],
         "args": json.loads(tc["function"]["arguments"] or "{}")}
        for tc in (msg.get("tool_calls") or [])
    ]
    return msg.get("content", ""), tool_calls, usage


def embed(texts):
    """Embeddings батчем от LM Studio. Возвращает список векторов"""
    r = requests.post(
        f"{BASE_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


class RunLog:
    """формат - одна строка JSON на событие"""

    def __init__(self, name):
        LOGS.mkdir(exist_ok=True)
        self.path = LOGS / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}-{name}.log"
        self.f = open(self.path, "w")

    def write(self, line):
        self.f.write(line if line.endswith("\n") else line + "\n")
        self.f.flush()

    def event(self, **kw):
        self.write(json.dumps(kw, ensure_ascii=False))

    def close(self):
        self.f.close()
