"""Мультитерн-диалог с историей, summarization и checkpoint/resume

Токен-бюджет из ДЗ-1: при 8K входа TTFT ~20с, при 16К ~42с.
Поэтому HISTORY_TOKEN_LIMIT ~2500 токенов: старые ходы
сжимаются в summary, свежие ходы идут дословно, места хватает на чанки
поиска и ответ
"""
import json
import sys
from pathlib import Path

from common import chat, RunLog, ROOT

SESSIONS = ROOT / "sessions"
SESSIONS.mkdir(exist_ok=True)

HISTORY_CHAR_LIMIT = 10000  # ~2500-3000 токенов
SUMMARY_PROMPT = (
    "Summarize the following dialogue compactly, keeping every fact important "
    "for continuing the conversation (names, numbers, decisions, sources). "
    "Max 10 sentences.\n\n"
)


class Session:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            state = json.loads(self.path.read_text())
            self.summary = state["summary"]
            self.history = state["history"]
        else:
            self.summary = ""
            self.history = []

    def _build_messages(self):
        msgs = []
        convo = ""
        if self.summary:
            msgs.append({"role": "system", "content": f"Conversation so far (summary): {self.summary}"})
        for h in self.history:
            convo += f"{h['role']}: {h['content']}\n"
        msgs.append({"role": "user", "content": convo + "Continue the dialogue. Answer the last user message."})
        return msgs

    def summarize_old(self, log=None):
        convo = "\n".join(f"{h['role']}: {h['content']}" for h in self.history)
        if len(convo) <= HISTORY_CHAR_LIMIT:
            return
        summary, _, _ = chat([{"role": "user", "content": SUMMARY_PROMPT + convo}],
                             max_tokens=1536, log=log)
        self.summary = (self.summary + " " + summary).strip()
        self.history = []
        self.save()

    def ask(self, question, log=None):
        self.history.append({"role": "user", "content": question})
        answer, _, usage = chat(self._build_messages(), max_tokens=1536, log=log)
        self.history.append({"role": "assistant", "content": answer})
        if log:
            log.event(event="turn_done", question=question,
                      prompt_tokens=usage.get("prompt_tokens"),
                      completion_tokens=usage.get("completion_tokens"),
                      summary_len=len(self.summary))
        self.summarize_old(log)
        self.save()
        return answer

    def save(self):
        self.path.write_text(json.dumps(
            {"summary": self.summary, "history": self.history}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: memory_session.py <session.json> ['ask' question | 'show']")
    sess = Session(SESSIONS / sys.argv[1])
    cmd = sys.argv[2] if len(sys.argv) > 2 else "show"
    if cmd == "ask":
        log = RunLog("session")
        print(sess.ask(" ".join(sys.argv[3:]), log=log))
        log.close()
    else:
        print(f"summary: {sess.summary}")
        for h in sess.history:
            print(f"{h['role']}: {h['content']}")
