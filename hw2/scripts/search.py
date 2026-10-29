"""SEarch: загрузка индекса с диска и косинусный top-k"""
import json

import numpy as np

from common import INDEX_DIR, TOP_K_DEFAULT

_IDX = None


def _load():
    global _IDX
    if _IDX is None:
        _IDX = (
            np.load(INDEX_DIR / "vectors.npy"),
            json.loads((INDEX_DIR / "chunks.json").read_text()),
            json.loads((INDEX_DIR / "meta.json").read_text()),
        )
    return _IDX


def search(query, k=TOP_K_DEFAULT):
    """Возвращает [{doc_id, source, chunk_idx, text, score}], отсортировано по убыванию"""
    from common import embed
    vecs, chunks, meta = _load()
    q = np.array(embed([query])[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = vecs @ q
    top = np.argsort(-scores)[:k]
    return [
        {"doc_id": meta[i]["doc_id"], "source": meta[i]["source"],
         "chunk_idx": meta[i]["chunk_idx"], "text": chunks[i],
         "score": float(scores[i])}
        for i in top
    ]


if __name__ == "__main__":
    import sys
    for r in search(" ".join(sys.argv[1:]), k=5):
        print(f"[{r['score']:.3f}] {r['doc_id']}#{r['chunk_idx']}")
        print(r['text'][:200])
        print("---")
