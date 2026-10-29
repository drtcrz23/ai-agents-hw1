"""Ingest: документы -> чанки -> embeddings -> индекс на диск

Чанкинг: по абзацам с накоплением до CHUNK_TOKENS (приближение: 1 токен ~ 3.5 символа), overlap 50 токенов.
Абзац длиннее лимита режется по границам предложений
Метаданные чанка: doc_id, chunk_idx, позиция в документе
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

from common import DOCS, INDEX_DIR, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, embed, RunLog

CHARS_PER_TOKEN = 3.5  # приближение для английского научного текста
CHUNK_CHARS = int(CHUNK_TOKENS * CHARS_PER_TOKEN)
OVERLAP_CHARS = int(CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN)


def read_doc(path):
    """Читает pdf/txt и возвращает plain text (титул + абстракт + тело)"""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            sys.exit("нужен pypdf: pip install pypdf")
        reader = PdfReader(path)
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n\n".join(pages), len(reader.pages)
    text = path.read_text(errors="ignore")
    return text, None


def split_long_para(p):
    """Абзац длиннее лимита режется по границам предложений на куски <= CHUNK_CHARS"""
    if len(p) <= CHUNK_CHARS:
        return [p]
    sentences = re.split(r"(?<=[.!?])\s+", p)
    pieces, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > CHUNK_CHARS:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_text(text):
    """Разбивка по абзацам, склейка в чанки ~CHUNK_TOKENS с overlap"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    paras = [piece for p in paras for piece in split_long_para(p)]
    chunks = []
    cur = []
    cur_len = 0
    for p in paras:
        plen = len(p)
        if cur_len + plen > CHUNK_CHARS and cur:
            joined = "\n\n".join(cur)
            chunks.append(joined)
            tail = joined[-OVERLAP_CHARS:]
            cur = [tail]
            cur_len = len(tail)
        cur.append(p)
        cur_len += plen
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def main():
    log = RunLog("ingest")
    INDEX_DIR.mkdir(exist_ok=True)
    all_chunks, all_meta = [], []
    doc_ids = {}
    for path in sorted(DOCS.iterdir()):
        if path.suffix.lower() not in (".pdf", ".txt"):
            continue
        text, npages = read_doc(path)
        chunks = chunk_text(text)
        if not chunks:
            log.event(event="skip", file=path.name, reason="empty")
            continue
        doc_id = f"{path.stem}"[:40]
        doc_ids[path.name] = len(chunks)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_meta.append({
                "doc_id": doc_id,
                "source": path.name,
                "chunk_idx": i,
                "n_chunks": len(chunks),
                "n_pages": npages,
            })
        log.event(event="doc", file=path.name, chunks=len(chunks), chars=len(text))

    log.event(event="total", docs=len(doc_ids), chunks=len(all_chunks))
    vecs = []
    for i in range(0, len(all_chunks), 32):
        vecs.extend(embed(all_chunks[i:i + 32]))
        log.event(event="embed_batch", done=min(i + 32, len(all_chunks)))

    mat = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    np.save(INDEX_DIR / "vectors.npy", mat / norms)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False))
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(all_meta, ensure_ascii=False))
    log.event(event="saved", dim=mat.shape)
    log.close()
    print(f"docs={len(doc_ids)} chunks={len(all_chunks)} -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
