"""Agentic: модель сама вызывает search(query). Цикл tool-calling до ответа"""
import json
import sys

from common import chat, RunLog, TOP_K_DEFAULT
from search import search

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Full-text search over a corpus of scientific papers. "
                       "Returns top-k relevant text chunks with source file names.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Search query in English"},
                "k": {"type": "integer", "description": "top-k, default 5"}
            },
            "required": ["query"],
        },
    },
}]

SYSTEM = (
    "You are a research assistant with access to a search tool over a corpus "
    "of scientific papers. Before answering, ALWAYS make at least TWO search "
    "calls (for comparison questions, search once per system/topic). "
    "Answer using ONLY retrieved context. Keep the answer under 250 words. "
    "End with a final line 'Sources: <comma-separated source file names you actually used>'."
)

MIN_SEARCHES = 2
MAX_TURNS = 6


def run_tool(name, args):
    if name != "search":
        return json.dumps({"error": f"unknown tool {name}"})
    hits = search(args["query"], k=args.get("k", TOP_K_DEFAULT))
    return json.dumps([
        {"source": h["source"], "chunk_idx": h["chunk_idx"],
         "score": round(h["score"], 3), "text": h["text"]}
        for h in hits
    ], ensure_ascii=False)


def agentic_answer(question, log=None, max_turns=MAX_TURNS):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    total_prompt = total_completion = 0
    n_searches = 0
    trace = []
    for turn in range(max_turns):
        answer, tool_calls, usage = chat(messages, tools=TOOLS,
                                         max_tokens=1536, log=log)
        total_prompt += usage.get("prompt_tokens", 0) or 0
        total_completion += usage.get("completion_tokens", 0) or 0
        if log:
            log.event(event="turn", turn=turn, content=answer[:200],
                      tool_calls=[(tc["name"], tc["args"]) for tc in tool_calls])
        if not tool_calls:
            if n_searches < MIN_SEARCHES:
                # nudge: модель хочет ответить раньше двух поисков - добираем вторым запросом
                if log:
                    log.event(event="nudge", turn=turn, n_searches=n_searches)
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content":
                    "Before answering, make at least one more search call with a "
                    "DIFFERENT query covering another aspect of the question."})
                continue
            trace.append({"turn": turn, "type": "final_answer", "content": answer})
            break
        messages.append({"role": "assistant", "content": answer,
                         "tool_calls": [
                             {"id": tc["id"], "type": "function",
                              "function": {"name": tc["name"],
                                            "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                             for tc in tool_calls]})
        for tc in tool_calls:
            n_searches += 1
            if log:
                log.event(event="tool_call", turn=turn, name=tc["name"], args=tc["args"])
            result = run_tool(tc["name"], tc["args"])
            if log:
                log.event(event="tool_result", turn=turn, call_id=tc["id"],
                          result=result)
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": result})
            trace.append({"turn": turn, "type": "search", "query": tc["args"].get("query"),
                          "call_id": tc["id"]})
    else:
        # модель не остановилась за max_turns - забираем последний ответ без tools
        messages.append({"role": "user", "content": "Now give a final answer without any tool calls."})
        answer, _, usage = chat(messages, max_tokens=1536, log=log)
        total_prompt += usage.get("prompt_tokens", 0) or 0
        total_completion += usage.get("completion_tokens", 0) or 0
        trace.append({"turn": max_turns, "type": "final_answer_forced", "content": answer})

    if log:
        log.event(event="answer", mode="agentic", question=question, answer=answer,
                  n_searches=n_searches, trace=trace,
                  prompt_tokens=total_prompt, completion_tokens=total_completion)
    return {"answer": answer, "n_searches": n_searches, "trace": trace,
            "usage": {"prompt_tokens": total_prompt, "completion_tokens": total_completion}}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("question: ")
    log = RunLog("agentic")
    r = agentic_answer(q, log=log)
    log.close()
    print(r["answer"])
    print(f"\n[searches: {r['n_searches']}]")
