#!/usr/bin/env python3
"""
Reproduction: claude-opus-4-8 returns literal <cite index="X-Y">...</cite> XML
tags in response text (with citations: null) instead of structured
search_result_location citations.

The failure needs FOUR ingredients at once (see README.md). The matrix below
removes one ingredient per arm, across three models:

    arms:
      trigger      — all four ingredients (fails ~25-40% on claude-opus-4-8)
      no-cite-task — same, but the prior tool_use task has no quote/cite language
      small-tools  — same, but the tool descriptions are ~2K chars instead of ~5K
    models: claude-opus-4-8, claude-opus-4-7, claude-sonnet-4-6

Only (trigger, claude-opus-4-8) fails. Every other cell is clean.

Usage:
    cp .env.example .env   # put your ANTHROPIC_API_KEY in .env
    pip install -r requirements.txt
    python reproduce.py                          # full 3x3 matrix, 12 attempts/cell
    python reproduce.py --attempts 6             # quicker pass
    python reproduce.py --models claude-opus-4-8 --arms trigger
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from anthropic import Anthropic

_HERE = os.path.dirname(os.path.abspath(__file__))

MAX_TOKENS = 16384
DEFAULT_MODELS = ["claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"]
ARMS = ["trigger", "no-cite-task", "small-tools"]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env file next to this script, if present."""
    path = os.path.join(_HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _load(name: str):
    with open(os.path.join(_HERE, name)) as f:
        return json.load(f)


@dataclass
class CellResult:
    failures: list[int] = field(default_factory=list)  # <cite> tags per failure
    cleans: list[int] = field(default_factory=list)  # structured citations per clean run
    other_tags: Counter = field(default_factory=Counter)  # any non-cite XML tags seen

    @property
    def total(self) -> int:
        return len(self.failures) + len(self.cleans)

    @property
    def rate(self) -> str:
        return f"{100 * len(self.failures) / self.total:.0f}%" if self.total else "n/a"


def build_request(data: dict, arm: str) -> dict:
    """Build the full messages.stream kwargs for one arm (model added later)."""
    task = data["clean_task"] if arm == "no-cite-task" else data["citation_language_task"]
    tool_input: dict = {"task": task}
    if arm != "no-cite-task":
        tool_input["thoroughness"] = "thorough"

    extra_tools = _load("tools_small.json" if arm == "small-tools" else "tools_large.json")

    messages = [
        {"role": "user", "content": [{"type": "text", "text": data["user_question"]}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tool_01", "name": "explore", "input": tool_input}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_01", "content": data["search_results"]}
            ]
            # The sub-agent's summary rides as a sibling text block — the API
            # rejects text blocks INSIDE a tool_result that contains
            # search_result blocks (all blocks must share that type).
            + ([{"type": "text", "text": data["result_summary"]}] if data.get("result_summary") else []),
        },
    ]

    return dict(
        max_tokens=MAX_TOKENS,
        system=data["system_prompt"],
        messages=messages,
        tools=[data["explore_tool"]] + extra_tools,
        # "none" keeps tool definitions in context but prevents tool calls, so
        # every attempt yields a citable answer. The failure reproduces at a
        # similar rate with {"type": "auto"} (production's setting).
        tool_choice={"type": "none"},
    )


# Failure detection is a literal "<cite" check: it catches complete tags,
# truncated tags, and counts openings only (matching how citations appear).
_CITE_OPEN_RE = re.compile(r"<cite\b")
# Census of any other XML-like tags — informational only, never drives
# pass/fail. Across all our testing, <cite> is the only tag ever observed.
_TAG_RE = re.compile(r"</?([a-zA-Z_][a-zA-Z0-9_-]*)(?:\s[^>]*)?>")


def run_attempt(client: Anthropic, data: dict, model: str, arm: str):
    """Returns (failed: bool, count: int, other_tags: Counter).

    failed=True when the response text contains literal <cite> tags
    (count = number of tags). Otherwise count = structured
    search_result_location citations. other_tags reports any additional
    XML-like tags in the text (none expected; flagged if seen).
    """
    with client.messages.stream(model=model, **build_request(data, arm)) as stream:
        response = stream.get_final_message()

    text = "".join(b.text or "" for b in response.content if b.type == "text")
    cite_tags = len(_CITE_OPEN_RE.findall(text))
    tags = Counter(m.group(1) for m in _TAG_RE.finditer(text))
    tags.pop("cite", None)

    if cite_tags > 0:
        return (True, cite_tags, tags)

    structured = sum(
        1
        for b in response.content
        if b.type == "text" and b.citations
        for c in b.citations
        if c.type == "search_result_location"
    )
    return (False, structured, tags)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=12, help="attempts per cell (default 12)")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--data",
        default="data.json",
        help="payload file (data.json = original-length results; "
        "data-small.json = short clean results)",
    )
    args = parser.parse_args()

    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY (or put it in .env — see .env.example)", file=sys.stderr)
        return 2

    data = _load(args.data)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    client = Anthropic()

    cells = {(m, a): CellResult() for m in models for a in arms}
    jobs = [(m, a, i) for m in models for a in arms for i in range(1, args.attempts + 1)]
    done = 0

    log(f"Running {len(jobs)} requests "
        f"({len(models)} models x {len(arms)} arms x {args.attempts} attempts, "
        f"concurrency={args.concurrency})...\n")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_attempt, client, data, m, a): (m, a, i) for m, a, i in jobs}
        for fut in as_completed(futures):
            model, arm, i = futures[fut]
            try:
                failed, count, other_tags = fut.result()
            except Exception as e:  # transient API errors: skip the attempt, keep the matrix
                done += 1
                log(f"[{done:>3}/{len(jobs)}] {model:<18} {arm:<14} attempt {i:>2}: "
                    f"API error, skipped ({type(e).__name__})")
                continue
            cell = cells[(model, arm)]
            if failed:
                cell.failures.append(count)
                detail = f"FAIL  ({count} literal <cite> tags, 0 structured citations)"
            else:
                cell.cleans.append(count)
                detail = f"clean ({count} structured citations)"
            if other_tags:
                cell.other_tags.update(other_tags)
                detail += f"  [UNEXPECTED XML TAGS: {dict(other_tags)}]"
            done += 1
            log(f"[{done:>3}/{len(jobs)}] {model:<18} {arm:<14} attempt {i:>2}: {detail}")

    col1 = max(len(m) for m in models) + 2
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"{'model':<{col1}}{'arm':<16}{'failures':<12}{'failure rate':<14}")
    print("-" * 72)
    for m in models:
        for a in arms:
            cell = cells[(m, a)]
            print(f"{m:<{col1}}{a:<16}{f'{len(cell.failures)}/{cell.total}':<12}{cell.rate:<14}")
    print("-" * 72)
    print('fail = response text contains literal <cite index="X-Y">...</cite> tags\n'
          "       and zero structured (search_result_location) citations.")
    seen_other = Counter()
    for cell in cells.values():
        seen_other.update(cell.other_tags)
    if seen_other:
        print(f"UNEXPECTED XML tags seen (beyond <cite>): {dict(seen_other)}")

    return 1 if any(c.failures for c in cells.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
