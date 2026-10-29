"""Baseline: вопрос -> один поиск -> чанки в промпт -> ответ"""
import json
import sys

from common import chat, RunLog, TOP_K_DEFAULT
from search import search

SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the provided "
    "context chunks from scientific papers. Keep the answer under 250 words. "
    "End with a final line 'Sources: <comma-separated source file names you actually used>'. "
    "If the context is insufficient, say so."
)


def baseline_answer(question, k=TOP_K_DEFAULT, log=None):
    hits = search(question, k=k)
    if log:
        log.event(event="search", mode="baseline", query=question, k=k,
                 hits=[(h["doc_id"], h["chunk_idx"], round(h["score"], 3)) for h in hits])
    ctx = "\n\n---\n\n".join(
        f"[{h['source']} chunk {h['chunk_idx']}]\n{h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx}"},
    ]
    answer, _, usage = chat(messages, max_tokens=1536, log=log)
    if log:
        log.event(event="answer", mode="baseline", question=question,
                  answer=answer, prompt_tokens=usage.get("prompt_tokens"),
                  completion_tokens=usage.get("completion_tokens"))
    return {"answer": answer, "hits": hits, "usage": usage}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("question: ")
    log = RunLog("baseline")
    r = baseline_answer(q, log=log)
    log.close()
    print(r["answer"])
