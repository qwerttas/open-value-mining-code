#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")

ATTRIBUTES = [
    "goal",
    "principle",
    "tradeoff",
    "beneficiaries",
    "harmed_groups",
    "norm",
    "perspective",
]

DIMENSION_DEFINITIONS = """Dimension definitions:
- goal: the desired outcome or state of affairs that the argument supports.
- principle: the normative reason that makes the goal desirable, legitimate, or necessary.
- tradeoff: the competing value, cost, risk, or sacrifice involved in accepting the conclusion.
- beneficiaries: the people, groups, institutions, or entities expected to benefit.
- harmed_groups: the people, groups, institutions, or entities expected to be harmed, burdened, or excluded.
- norm: the behavioral expectation, duty, or social rule implied by the argument.
- perspective: the evaluative standpoint from which the argument judges the issue.
"""

PROMPT = """You are given one argument edge in context.

Task: describe the value expressed by the Premise-Conclusion-Stance edge using seven semantic dimensions. Each value must be exactly one concise English sentence.

{dimension_definitions}

Paragraph:
{paragraph}

Premise:
{premise}

Conclusion:
{conclusion}

Relation: {relation}
Stance: {stance}

Return JSON only:
{{
  "goal": "...",
  "principle": "...",
  "tradeoff": "...",
  "beneficiaries": "...",
  "harmed_groups": "...",
  "norm": "...",
  "perspective": "..."
}}

Rules:
- Ground every field in the argument and paragraph context.
- If a field is not specified, write "Not specified in the paragraph."
- Do not start with "This sentence" or "The unit".
- Output JSON only.
"""


def client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for value decomposition.")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    return text[start:].strip() if start >= 0 else text


def call_json(prompt: str, temperature: float, timeout: int, retries: int) -> dict[str, Any]:
    llm = client()
    last_error = None
    for attempt in range(retries):
        try:
            resp = llm.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            obj = json.loads(clean_json(resp.choices[0].message.content or ""))
            if isinstance(obj, dict):
                return obj
            last_error = "non_object_json"
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM JSON call failed: {last_error}")


def valid_attributes(obj: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key in ATTRIBUTES:
        value = str(obj.get(key) or "").strip()
        if key == "goal" and not value:
            value = str(obj.get("final_goal") or "").strip()
        if key == "tradeoff" and not value:
            value = str(obj.get("trade_off") or "").strip()
        if not value:
            value = "Not specified in the paragraph."
        out[key] = value
    return out


def iter_edges(row: dict[str, Any]) -> list[dict[str, Any]]:
    paragraph = str(row.get("paragraph_text") or row.get("text") or "").strip()
    pack_id = str(row.get("pack_id") or row.get("id") or "")
    units = {str(u.get("unit_id")): str(u.get("text") or "") for u in row.get("units_pred") or row.get("units") or [] if isinstance(u, dict)}
    pairs = row.get("pairs_pred") or row.get("pairs") or []
    edges = []
    for idx, edge in enumerate(pairs, 1):
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src_unit_id") or edge.get("premise_unit_id") or "").strip()
        tgt = str(edge.get("tgt_unit_id") or edge.get("conclusion_unit_id") or "").strip()
        premise = str(edge.get("premise_text") or units.get(src) or "").strip()
        conclusion = str(edge.get("conclusion_text") or units.get(tgt) or "").strip()
        if not premise or not conclusion:
            continue
        edges.append({
            "pack_id": pack_id,
            "edge_id": str(edge.get("edge_id") or f"e{idx}"),
            "src_unit_id": src,
            "tgt_unit_id": tgt,
            "premise_text": premise,
            "conclusion_text": conclusion,
            "relation": str(edge.get("relation") or "support").strip(),
            "stance": str(edge.get("stance") or "in favor of").strip(),
            "paragraph_text": paragraph,
        })
    return edges


def build_prompt(edge: dict[str, Any]) -> str:
    return PROMPT.format(
        dimension_definitions=DIMENSION_DEFINITIONS,
        paragraph=edge.get("paragraph_text", ""),
        premise=edge.get("premise_text", ""),
        conclusion=edge.get("conclusion_text", ""),
        relation=edge.get("relation", ""),
        stance=edge.get("stance", ""),
    )


def decompose(edge: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    obj = call_json(build_prompt(edge), args.temperature, args.timeout, args.retries)
    out = dict(edge)
    out["attributes"] = valid_attributes(obj)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate value-semantic attributes for P-C-S edges.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--failed-jsonl", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    output = Path(args.output_jsonl)
    failed = Path(args.failed_jsonl) if args.failed_jsonl else output.with_suffix(".failed.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    failed.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for row in read_jsonl(Path(args.input_jsonl)):
        tasks.extend(iter_edges(row))

    results: list[Optional[dict[str, Any]]] = [None] * len(tasks)
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(decompose, edge, args): (idx, edge) for idx, edge in enumerate(tasks)}
        for future in as_completed(futures):
            idx, edge = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                failures.append({"source": edge, "error": str(exc)})

    write_jsonl(output, [row for row in results if row is not None])
    write_jsonl(failed, failures)


if __name__ == "__main__":
    main()
