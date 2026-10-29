"""Оценка: recall@k индекса + agentic vs baseline на золотом наборе

Golden set: data/golden/golden_set.json -
[{"id": 1, "question": "...", "expected_docs": ["file1", "file2"],
  "multihop": true, "expected_facts": ["..."]}]
(expected_docs = stems имен файлов, multihop вопросы требуют 2+ документов)
"""
import json
import re
import sys

import numpy as np

from common import GOLDEN, RESULTS, RunLog, TOP_K_DEFAULT
from search import search
from baseline import baseline_answer
from agentic import agentic_answer


def recall_at_k(golden, ks=(1, 3, 5)):
    rows = []
    for item in golden:
        hits = search(item["question"], k=max(ks))
        hit_docs = [h["doc_id"] for h in hits]
        row = {"id": item["id"], "multihop": item["multihop"]}
        for k in ks:
            got = hit_docs[:k]
            row[f"recall@{k}"] = len(set(got) & set(item["expected_docs"])) / len(set(item["expected_docs"]))
        rows.append(row)
        print(row)
    table = {}
    for k in ks:
        vals = [r[f"recall@{k}"] for r in rows]
        table[f"recall@{k}"] = float(np.mean(vals))
        mh = [r[f"recall@{k}"] for r in rows if r["multihop"]]
        if mh:
            table[f"recall@{k}_multihop"] = float(np.mean(mh))
    return table, rows


SOURCE_RE = re.compile(r"Sources?:\s*(.+)", re.IGNORECASE)


def cited_docs(answer):
    m = SOURCE_RE.search(answer)
    if not m:
        return []
    return [s.strip().strip(".,") for s in re.split(r"[,;]", m.group(1)) if s.strip()]


def doc_match(cited, expected):
    """Небрежное сравнение: пересечение по подстрокам имен"""
    matched = set()
    for c in cited:
        for e in expected:
            if e.lower() in c.lower() or c.lower() in e.lower():
                matched.add(e)
    return matched


def main():
    golden = json.loads((GOLDEN / "golden_set.json").read_text())
    log = RunLog("eval")
    RESULTS.mkdir(exist_ok=True)

    rec_table, rec_rows = recall_at_k(golden)
    print("recall:", rec_table)
    log.event(event="recall_table", **rec_table)

    comparisons = []
    for item in golden:
        b = baseline_answer(item["question"], log=log)
        a = agentic_answer(item["question"], log=log)
        row = {
            "id": item["id"],
            "multihop": item["multihop"],
            # качество: доля expected_docs, названных в Sources
            "baseline_docs": sorted(doc_match(cited_docs(b["answer"]), item["expected_docs"])),
            "agentic_docs": sorted(doc_match(cited_docs(a["answer"]), item["expected_docs"])),
            # стоимость
            "baseline_tokens": (b["usage"].get("prompt_tokens", 0),
                                b["usage"].get("completion_tokens", 0)),
            "agentic_tokens": (a["usage"]["prompt_tokens"], a["usage"]["completion_tokens"]),
            "baseline_searches": 1,
            "agentic_searches": a["n_searches"],
        }
        row["baseline_score"] = len(row["baseline_docs"]) / len(item["expected_docs"])
        row["agentic_score"] = len(row["agentic_docs"]) / len(item["expected_docs"])
        comparisons.append(row)
        log.event(event="comparison", **row)

    def agg(rows, key):
        return {
            "mean": float(np.mean([r[key] for r in rows])),
            "multihop_mean": float(np.mean([r[key] for r in rows if r["multihop"]])),
            "single_hop_mean": float(np.mean([r[key] for r in rows if not r["multihop"]])),
        }

    summary = {
        "n_questions": len(golden),
        "recall": rec_table,
        "baseline_doc_accuracy": agg(comparisons, "baseline_score"),
        "agentic_doc_accuracy": agg(comparisons, "agentic_score"),
        "baseline_tokens_per_q": agg_rows(comparisons, "baseline_tokens"),
        "agentic_tokens_per_q": agg_rows(comparisons, "agentic_tokens"),
        "baseline_searches_per_q": float(np.mean([r["baseline_searches"] for r in comparisons])),
        "agentic_searches_per_q": float(np.mean([r["agentic_searches"] for r in comparisons])),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (RESULTS / "eval.json").write_text(json.dumps(
        {"summary": summary, "comparisons": comparisons,
         "recall_rows": rec_rows}, indent=2, ensure_ascii=False))
    log.event(event="summary", **summary)
    log.close()


def agg_rows(rows, key):
    p = float(np.mean([r[key][0] for r in rows]))
    c = float(np.mean([r[key][1] for r in rows]))
    return {"prompt": p, "completion": c, "total": p + c}


if __name__ == "__main__":
    main()
