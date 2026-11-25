"""Coding-агент: ReAct-цикл (действие -> результат -> исправление) над tasks/<задача>/

Запуск: python run_agent.py --task tasks/puzzle-8
Артефакты в папке задачи: experiment.py (пишет агент), train.log (лог эксперимента),
agent.log (пошаговый лог агента), REPORT.md (отчёт агента).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common import chat

ROOT = Path(__file__).parent

# Ограничения (задание: max agent steps + run timeout)
MAX_STEPS = 35
RUN_TIMEOUT = 120        # сек на один запуск run_python
TOTAL_TIMEOUT = 30 * 60  # потолок на всю работу агента
MAX_OUTPUT_CHARS = 3500  # обрезка вывода эксперимента для контекста модели
MAX_READ_CHARS = 8000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the task directory. Returns the file content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "path relative to the task dir"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a file in the task directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a Python file inside the task directory. "
                           "Enforced timeout. Returns exit code, stdout/stderr, "
                           "and the metric verdict against expected.json when METRIC "
                           "lines are printed.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

SYSTEM = """You are a coding agent solving an experiment task in a task directory.

Workflow (ReAct — act, observe the result, fix, repeat):
1. Read task.md and expected.json.
2. Write experiment.py with the write_file tool.
3. Run it with run_python. If it fails or times out, read the error, fix the code and run again. Repeat until it succeeds.
4. When the experiment succeeds, check the printed METRIC lines against the expected range in expected.json (the tool result also shows the verdict).
5. Write REPORT.md via write_file: approach, algorithm, metric values, in-range check, and which errors you fixed along the way.
6. Reply with a short final summary.

Rules:
- Paths are relative to the task directory; you cannot escape it.
- Use only the Python stdlib in experiment.py.
- When a run fails or produces zero metrics, do NOT rewrite the whole experiment
  blindly. Write a small debug script that tests the solver on a minimal case
  (the goal state itself, then a board one move from the goal), run it, and fix
  the specific bug it reveals.
- Do not claim success before experiment.py runs cleanly and the metric is inside the expected range and REPORT.md is written.
"""


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class AgentLog:
    def __init__(self, path):
        self.path = path
        self.path.write_text("")  # каждая сессия начинает лог заново

    def append(self, text):
        with self.path.open("a") as f:
            f.write(text.rstrip("\n") + "\n")


def safe_path(task_dir, rel):
    """Путь обязан лежать внутри task_dir (containment через parents,
    а не префикс строки: 'puzzle-8-evil' не должен проходить проверку)."""
    p = (task_dir / rel).resolve()
    if p != task_dir and task_dir not in p.parents:
        raise ValueError(f"path escapes task dir: {rel}")
    return p


def parse_metrics(stdout):
    """Достаёт METRIC-строки вида 'METRIC name value'"""
    metrics = {}
    for m in re.finditer(r"METRIC\s+(\w+)\s+([0-9.eE+-]+)", stdout):
        metrics[m.group(1)] = float(m.group(2))
    return metrics


def metric_verdict(stdout, expected):
    metrics = parse_metrics(stdout)
    exp = json.loads(expected)
    name, lo, hi = exp["metric"], exp["min"], exp["max"]
    parts = []
    if name in metrics:
        ok = lo <= metrics[name] <= hi
        parts.append(f"{name}={metrics[name]:.4g}, expected [{lo}, {hi}] -> {'PASS' if ok else 'FAIL'}")
    else:
        parts.append(f"{name} missing -> FAIL (no 'METRIC {name} <value>' line in stdout)")
    if "solved_rate" in metrics:
        sr_ok = metrics["solved_rate"] == 1.0
        parts.append(f"solved_rate={metrics['solved_rate']:.4g} -> {'PASS' if sr_ok else 'FAIL'} (must be 1.0)")
    else:
        parts.append("solved_rate missing -> FAIL (no 'METRIC solved_rate <value>' line in stdout)")
    return metrics, "; ".join(parts)


def truncate(text, head=600, tail=2500):
    if len(text) <= head + tail + 50:
        return text
    return f"{text[:head]}\n... [truncated {len(text) - head - tail} chars] ...\n{text[-tail:]}"


def run_tool(name, args_json, task_dir, expected_raw, alog, step):
    arg_preview = truncate(args_json, head=0, tail=700)
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError as e:
        alog.append(f"[{ts()}] step {step} | tool={name} | args={arg_preview} | error=bad json: {e}")
        return "error", f"invalid tool arguments (bad JSON): {e}", False

    if name == "read_file":
        try:
            p = safe_path(task_dir, args["path"])
            content = p.read_text(errors="replace")
            alog.append(f"[{ts()}] step {step} | tool=read_file | args={args_json} | result=ok ({len(content)} chars)")
            return "ok", content[:MAX_READ_CHARS], False
        except Exception as e:
            alog.append(f"[{ts()}] step {step} | tool=read_file | args={args_json} | error={type(e).__name__}: {e}")
            return "error", f"error: {type(e).__name__}: {e}", False

    if name == "write_file":
        try:
            p = safe_path(task_dir, args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and p.read_text() == args["content"]:
                alog.append(f"[{ts()}] step {step} | tool=write_file | args={json.dumps({'path': args['path']})} | error=identical content, nothing changed")
                return "identical", (
                    f"The file {args['path']} already contains EXACTLY the content you "
                    f"wrote. Rewriting the same code will not change the result. "
                    f"Change your approach, do not repeat the same code."), False
            p.write_text(args["content"])
            alog.append(f"[{ts()}] step {step} | tool=write_file | args={json.dumps({'path': args['path']})} | result=ok ({len(args['content'])} chars written)")
            return "ok", f"wrote {len(args['content'])} chars to {args['path']}", False
        except Exception as e:
            alog.append(f"[{ts()}] step {step} | tool=write_file | args={arg_preview} | error={type(e).__name__}: {e}")
            return "error", f"error: {type(e).__name__}: {e}", False

    if name == "run_python":
        try:
            p = safe_path(task_dir, args["path"])
        except Exception as e:
            return "error", f"error: {e}", False
        if not p.exists():
            alog.append(f"[{ts()}] step {step} | tool=run_python | args={args_json} | error=file not found: {args['path']}")
            return "error", f"FileNotFoundError: {args['path']} (write it first)", False
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(p)], cwd=str(task_dir),
                capture_output=True, text=True, timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            alog.append(f"[{ts()}] step {step} | tool=run_python | args={args_json} | error=timeout after {RUN_TIMEOUT}s")
            return "error", f"TIMEOUT: the experiment exceeded the {RUN_TIMEOUT}s limit. Optimize the code.", False
        dur = time.time() - t0
        out = proc.stdout or ""
        err = proc.stderr or ""
        combined = out + (("\n[stderr]\n" + err) if err.strip() else "")
        alog.append(f"[{ts()}] step {step} | tool=run_python | args={args_json} | run record: exit={proc.returncode} time={dur:.1f}s")
        if proc.returncode != 0:
            alog.append(f"[{ts()}] step {step} | tool=run_python | args={args_json} | error=exit {proc.returncode} after {dur:.1f}s")
            return "error", f"exit code {proc.returncode} after {dur:.1f}s\n{truncate(combined)}", False
        metrics, verdict = metric_verdict(out, expected_raw)
        metrics_ok = "FAIL" not in (verdict or "")
        alog.append(f"[{ts()}] step {step} | tool=run_python | args={args_json} | result=ok exit=0 in {dur:.1f}s | {verdict or 'no METRIC lines'}")
        verdict_hint = f"\n\nMETRIC VERDICT: {verdict}" if verdict else ""
        if metrics.get("solved_rate", 1.0) == 0.0 or metrics.get("avg_moves", 1.0) == 0.0:
            verdict_hint += (
                "\n\nHINT: the solver returned no solutions at all. The bug is "
                "almost certainly in the goal check or in move generation. Do NOT "
                "rewrite the whole experiment: write a tiny debug script that "
                "tests the solver on the goal state itself (empty solution) and on "
                "a board one move from the goal (exactly one move), print the "
                "search trace, and fix the specific bug you see.")
        if metrics_ok:
            train_log = task_dir / "train.log"
            summary_ok = False
            if train_log.exists():
                non_empty = [l.strip() for l in train_log.read_text().splitlines() if l.strip()]
                summary_ok = bool(non_empty) and non_empty[-1].lower().startswith("summary")
            if not summary_ok:
                metrics_ok = False
                alog.append(f"[{ts()}] step {step} | tool=run_python | args={args_json} | SPEC VIOLATION: train.log does not end with a Summary line")
                verdict_hint += (
                    "\n\nSPEC VIOLATION: train.log must END with a final summary line "
                    "of the form 'Summary solved=<n>/100 avg_moves=<x>' as its LAST "
                    "line. The current train.log does not. Fix experiment.py to write "
                    "the summary after the loop and rerun.")
        return "ok", truncate(combined) + verdict_hint, metrics_ok

    return "error", f"unknown tool: {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task dir, e.g. tasks/puzzle-8")
    args = ap.parse_args()

    task_dir = (ROOT / args.task).resolve()
    task_md = (task_dir / "task.md").read_text()
    expected_raw = (task_dir / "expected.json").read_text()
    alog = AgentLog(task_dir / "agent.log")

    alog.append(f"[{ts()}] session start task={args.task} model=qwen/qwen3.5-9b max_steps={MAX_STEPS} run_timeout={RUN_TIMEOUT}s")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Solve the task in the task directory.\n\n"
            f"--- task.md ---\n{task_md}\n--- expected.json ---\n{expected_raw}\n"
            f"--- end ---\nBegin now.")},
    ]

    total_prompt = total_completion = 0
    metrics_pass = False  # вердикт последнего запуска эксперимента
    consecutive_identical = 0
    nudges = 0
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        if time.time() - t0 > TOTAL_TIMEOUT:
            alog.append(f"[{ts()}] TOTAL_TIMEOUT {TOTAL_TIMEOUT}s reached, stopping")
            break
        msg, usage = chat(messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": msg.get("tool_calls")})
        total_prompt += usage.get("prompt_tokens", 0)
        total_completion += usage.get("completion_tokens", 0)

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                args_json = tc["function"].get("arguments", "{}")
                status, result, ok = run_tool(name, args_json, task_dir, expected_raw, alog, step)
                if name == "run_python" and status == "ok":
                    metrics_pass = ok
                if status == "identical":
                    consecutive_identical += 1
                elif name == "write_file" and status == "ok":
                    # сбрасывать только когда write реально изменил файл,
                    # а не на каждый run_read/run_python между ними
                    consecutive_identical = 0
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            if consecutive_identical >= 3:
                consecutive_identical = 0
                messages.append({"role": "user", "content":
                    "You tried to write the identical file three times in a row. "
                    "STOP rewriting. Change strategy: write a minimal debug script "
                    "(debug.py) that tests the solver on the goal state itself and on "
                    "a board one move from the goal, prints every state it expands, "
                    "run it, and fix the specific bug it reveals."})
        else:
            text = (msg.get("content") or "").strip()
            alog.append(f"[{ts()}] step {step} | message | {text[:300]}")
            report_ok = (task_dir / "REPORT.md").exists()
            if report_ok and metrics_pass:
                alog.append(f"[{ts()}] done: REPORT.md written and metrics PASS, total steps={step}")
                break
            nudges += 1
            if nudges > 2:
                alog.append(f"[{ts()}] stop: no tool calls after {nudges} nudges (metrics_pass={metrics_pass}, REPORT.md={report_ok})")
                break
            if not metrics_pass:
                messages.append({"role": "user", "content":
                    "The metrics are not verified as PASS yet. Fix the experiment until "
                    "the METRIC verdict is PASS, then write REPORT.md."})
            else:
                messages.append({"role": "user", "content":
                    "Metrics already PASS: write REPORT.md now (if not yet) and finish."})
    else:
        alog.append(f"[{ts()}] stop: MAX_STEPS={MAX_STEPS} reached")

    alog.append(f"[{ts()}] session end: steps used, tokens prompt={total_prompt} completion={total_completion}, wall={time.time() - t0:.0f}s")
    print(f"tokens: prompt={total_prompt} completion={total_completion}")
    print(f"artifacts in {task_dir}")


if __name__ == "__main__":
    main()
