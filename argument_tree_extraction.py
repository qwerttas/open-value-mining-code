#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")

STEP1_PROMPT = """You are an argument structure annotator.

Task: Split the paragraph into minimal argumentative units and mark conclusion units.
The paragraph may be written in any language.

Paragraph:
{paragraph}

Return JSON only:
{{
  "units": [{{"unit_id": "u1", "text": "..."}}],
  "conclusion_unit_ids": ["u2"]
}}

Rules:
- Preserve the original language.
- Unit text must be copied from the paragraph, not translated or paraphrased.
- Use unit_id values u1, u2, u3, ...
- Conclusion units include root and intermediate conclusions.
- Output JSON only.
"""

STEP2_PROMPT = """You are an argument structure annotator.

Task: Select the root conclusion unit(s) from the conclusion units.

Paragraph:
{paragraph}

Units:
{units}

Conclusion unit IDs:
{conclusion_ids}

Return JSON only:
{{"root_unit_ids": ["u2"]}}

Rules:
- root_unit_ids must be a non-empty subset of conclusion_unit_ids.
- Select the most central final conclusion(s).
- Output JSON only.
"""

STEP3_PROMPT = """You are an argument structure annotator.

Task: For each target conclusion, find premise units that directly support or attack it.

Paragraph:
{paragraph}

Units:
{units}

Conclusion unit IDs:
{conclusion_ids}

Target conclusion unit IDs:
{target_ids}

Return JSON only:
{{
  "expansions": [
    {{
      "target_unit_id": "u2",
      "premises": [
        {{"unit_id": "u3", "relation": "support", "stance": "in favor of"}}
      ]
    }}
  ]
}}

Rules:
- Premise IDs must come from the unit list.
- relation must be support or attack.
- stance must be in favor of or against and must match relation.
- Add a link only when the paragraph gives explicit support or contrast cues.
- Use 0-3 premises for each target.
- Output JSON only.
"""


def client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for extraction.")
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
    start = min([i for i in [text.find("{"), text.find("[")] if i >= 0], default=0)
    return text[start:].strip()


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
            content = resp.choices[0].message.content or ""
            obj = json.loads(clean_json(content))
            if isinstance(obj, dict):
                return obj
            last_error = "non_object_json"
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM JSON call failed: {last_error}")


def normalize_units(raw: Any) -> list[dict[str, str]]:
    units = []
    seen = set()
    if not isinstance(raw, list):
        return units
    for idx, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        unit_id = str(item.get("unit_id") or f"u{idx}").strip()
        if unit_id in seen:
            unit_id = f"u{idx}"
        seen.add(unit_id)
        units.append({"unit_id": unit_id, "text": text})
    return units


def keep_known(ids: Any, allowed: set[str]) -> list[str]:
    if not isinstance(ids, list):
        return []
    out = []
    for value in ids:
        unit_id = str(value).strip()
        if unit_id in allowed and unit_id not in out:
            out.append(unit_id)
    return out


def extract_units(paragraph: str, temperature: float, timeout: int, retries: int) -> tuple[list[dict[str, str]], list[str]]:
    obj = call_json(STEP1_PROMPT.format(paragraph=paragraph), temperature, timeout, retries)
    units = normalize_units(obj.get("units"))
    allowed = {u["unit_id"] for u in units}
    conclusion_ids = keep_known(obj.get("conclusion_unit_ids"), allowed)
    return units, conclusion_ids


def select_roots(paragraph: str, units: list[dict[str, str]], conclusion_ids: list[str], temperature: float, timeout: int, retries: int) -> list[str]:
    if not conclusion_ids and units:
        conclusion_ids = [units[-1]["unit_id"]]
    prompt = STEP2_PROMPT.format(
        paragraph=paragraph,
        units=json.dumps(units, ensure_ascii=False),
        conclusion_ids=json.dumps(conclusion_ids, ensure_ascii=False),
    )
    obj = call_json(prompt, temperature, timeout, retries)
    roots = keep_known(obj.get("root_unit_ids"), set(conclusion_ids))
    return roots or conclusion_ids[:1]


def expand_targets(
    paragraph: str,
    units: list[dict[str, str]],
    conclusion_ids: list[str],
    target_ids: list[str],
    temperature: float,
    timeout: int,
    retries: int,
) -> list[dict[str, str]]:
    prompt = STEP3_PROMPT.format(
        paragraph=paragraph,
        units=json.dumps(units, ensure_ascii=False),
        conclusion_ids=json.dumps(conclusion_ids, ensure_ascii=False),
        target_ids=json.dumps(target_ids, ensure_ascii=False),
    )
    obj = call_json(prompt, temperature, timeout, retries)
    allowed = {u["unit_id"] for u in units}
    edges = []
    expansions = obj.get("expansions")
    if not isinstance(expansions, list):
        return edges
    for expansion in expansions:
        if not isinstance(expansion, dict):
            continue
        tgt = str(expansion.get("target_unit_id") or "").strip()
        if tgt not in target_ids:
            continue
        premises = expansion.get("premises") or []
        if not isinstance(premises, list):
            continue
        for premise in premises[:3]:
            if not isinstance(premise, dict):
                continue
            src = str(premise.get("unit_id") or "").strip()
            relation = str(premise.get("relation") or "").strip().lower()
            stance = str(premise.get("stance") or "").strip().lower()
            if src not in allowed or src == tgt:
                continue
            if relation not in {"support", "attack"}:
                continue
            if stance not in {"in favor of", "against"}:
                stance = "in favor of" if relation == "support" else "against"
            edges.append({
                "src_unit_id": src,
                "tgt_unit_id": tgt,
                "relation": relation,
                "stance": stance,
            })
    return edges


def assemble_tree(
    paragraph: str,
    units: list[dict[str, str]],
    conclusion_ids: list[str],
    roots: list[str],
    temperature: float,
    timeout: int,
    retries: int,
    batch_size: int,
) -> list[dict[str, str]]:
    conclusion_set = set(conclusion_ids)
    seen_targets = set()
    seen_edges = set()
    edges: list[dict[str, str]] = []
    queue = deque(roots)
    while queue:
        batch = []
        while queue and len(batch) < batch_size:
            target = queue.popleft()
            if target not in seen_targets:
                seen_targets.add(target)
                batch.append(target)
        if not batch:
            continue
        for edge in expand_targets(paragraph, units, conclusion_ids, batch, temperature, timeout, retries):
            key = (edge["src_unit_id"], edge["tgt_unit_id"], edge["relation"], edge["stance"])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(edge)
            if edge["src_unit_id"] in conclusion_set and edge["src_unit_id"] not in seen_targets:
                queue.append(edge["src_unit_id"])
    return edges


def finalize(units: list[dict[str, str]], conclusion_ids: list[str], roots: list[str], edges: list[dict[str, str]]) -> dict[str, Any]:
    used = set(roots)
    for edge in edges:
        used.add(edge["src_unit_id"])
        used.add(edge["tgt_unit_id"])
    conclusion_set = set(conclusion_ids)
    root_set = set(roots)
    unit_map = {u["unit_id"]: u["text"] for u in units}
    out_units = []
    for unit in units:
        unit_id = unit["unit_id"]
        if unit_id not in used:
            continue
        role = "premise"
        if unit_id in root_set:
            role = "conclusion"
        elif unit_id in conclusion_set:
            role = "intermediate_conclusion"
        out_units.append({"unit_id": unit_id, "text": unit["text"], "role": role})
    pairs = []
    for idx, edge in enumerate(edges, 1):
        pairs.append({
            "edge_id": f"e{idx}",
            **edge,
            "premise_text": unit_map.get(edge["src_unit_id"], ""),
            "conclusion_text": unit_map.get(edge["tgt_unit_id"], ""),
        })
    return {"units_pred": out_units, "pairs_pred": pairs, "roots_pred": roots}


def extract_paragraph(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    paragraph = str(row.get(args.text_field) or row.get("paragraph_text") or row.get("text") or "").strip()
    if not paragraph:
        raise ValueError("missing paragraph text")
    units, conclusion_ids = extract_units(paragraph, args.temperature, args.timeout, args.retries)
    if not units:
        raise ValueError("no units extracted")
    roots = select_roots(paragraph, units, conclusion_ids, args.temperature, args.timeout, args.retries)
    edges = assemble_tree(paragraph, units, conclusion_ids, roots, args.temperature, args.timeout, args.retries, args.batch_size)
    out = dict(row)
    out["paragraph_text"] = paragraph
    out.update(finalize(units, conclusion_ids, roots, edges))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-constrained argument extraction.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--failed-jsonl", default=None)
    parser.add_argument("--text-field", default="paragraph_text")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    output = Path(args.output_jsonl)
    failed = Path(args.failed_jsonl) if args.failed_jsonl else output.with_suffix(".failed.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    failed.parent.mkdir(parents=True, exist_ok=True)

    results: list[Optional[dict[str, Any]]] = [None] * len(rows)
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_paragraph, row, args): (idx, row) for idx, row in enumerate(rows)}
        for future in as_completed(futures):
            idx, src = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                failures.append({"source": src, "error": str(exc)})

    write_jsonl(output, [row for row in results if row is not None])
    write_jsonl(failed, failures)


if __name__ == "__main__":
    main()
