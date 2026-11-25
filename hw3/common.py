import json
import time
import urllib.request

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3.5-9b"  # reasoning выключен

def chat(messages, tools=None, max_tokens=6000, temperature=0.2, retries=3):
    """Возвращает (message, usage). Retry на сетевые сбои; на finish_reason=length
    без tool_calls — повтор с удвоенным бюджетом."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools

    def _call(mtokens):
        payload = dict(body, max_tokens=mtokens)
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        return data

    for attempt in range(retries):
        try:
            data = _call(max_tokens)
            usage = data.get("usage", {})
            msg = data["choices"][0]["message"]
            if data["choices"][0].get("finish_reason") == "length" and not msg.get("tool_calls"):
                data = _call(max_tokens * 2)
                usage = {
                    k: usage.get(k, 0) + data.get("usage", {}).get(k, 0)
                    for k in ("prompt_tokens", "completion_tokens")
                }
                msg = data["choices"][0]["message"]
            return msg, usage
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)
