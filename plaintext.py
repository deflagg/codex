#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Recipelets JSONL to plain training text.

Usage examples:
  python convert_recipelets_to_text.py --in_dir ./recipelets_demo --out ./recipelets_corpus.txt --format tagged
  python convert_recipelets_to_text.py --in_dir ./recipelets --out ./corpus --shards 100 --format plain
  python convert_recipelets_to_text.py --in_dir ./recipelets --out ./corpus.txt.gz --format instruct --max_examples 100000

Formats:
- tagged:
    <unit=metric> <cuisine=italian> <diet=veg> <lvl=2> <method=saute>
    quick basil sauté
    ingredients:
    - 1 tbsp olive oil
    - 2 cloves garlic
    steps:
    1. Heat oil ...
    2. ...
- plain:
    quick basil sauté
    INGREDIENTS:
    - 1 tbsp olive oil
    ...
    STEPS:
    1. ...
- instruct:
    Write a <cuisine=italian> <method=saute> recipe for 2 servings under 15 minutes.
    ###
    quick basil sauté
    ingredients:
    - ...
    steps:
    1. ...
"""

import os, json, gzip, argparse, re, glob
from typing import Dict, Iterable, Tuple, List

def find_jsonl_files(in_dir: str) -> List[str]:
    pats = [os.path.join(in_dir, "recipelets_*.jsonl"),
            os.path.join(in_dir, "recipelets_*.jsonl.gz")]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No recipelets_*.jsonl(.gz) files in {in_dir}")
    return files

def open_maybe_gzip(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")

def ingredient_line(ing: Dict, unit_system: str) -> str:
    # mimic the generator's rendering rules briefly
    qty = ing["qty"]
    unit = ing["unit"]
    def pluralize(unit: str, qty) -> str:
        if unit in ("g","ml"):
            return unit
        if abs(float(qty) - 1.0) < 1e-9:
            return unit
        irr = {"clove":"cloves","leaf":"leaves","slice":"slices","piece":"pieces","egg":"eggs"}
        return irr.get(unit, unit+"s")
    unit_txt = pluralize(unit, qty)
    if isinstance(qty, float) and abs(qty - int(qty)) < 1e-9:
        qty_txt = str(int(qty))
    else:
        qty_txt = str(qty)
    # enforce metric label-only for metric unit system (no cups/tsp/tbsp strings)
    if unit_system == "metric" and unit in ("cup","tbsp","tsp"):
        unit = {"cup":"ml","tbsp":"ml","tsp":"ml"}[unit]
        unit_txt = unit
        try:
            qty_num = float(qty)
        except:
            qty_num = 1.0
        conv = 240 if unit == "ml" else 1
        qty_txt = str(max(5, int(qty_num*conv)))
    return f"- {qty_txt} {unit_txt} {ing['item']}"

def render_tagged(rec: Dict) -> str:
    t = rec["tags"]
    head = f"<unit={t['unit']}> <cuisine={t['cuisine']}> <diet={t['diet']}> <lvl={t['lvl']}> <method={t['method']}>"
    lines = [head, rec["title"], "ingredients:"]
    for ing in rec["ingredients"]:
        lines.append(ingredient_line(ing, t["unit"]))
    lines.append("steps:")
    for i, s in enumerate(rec["steps"], 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)

def render_plain(rec: Dict) -> str:
    t = rec["tags"]
    lines = [rec["title"], "INGREDIENTS:"]
    for ing in rec["ingredients"]:
        lines.append(ingredient_line(ing, t["unit"]))
    lines.append("STEPS:")
    for i, s in enumerate(rec["steps"], 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)

def render_instruct(rec: Dict) -> str:
    t = rec["tags"]
    prompt = f"Write a <cuisine={t['cuisine']}> <method={t['method']}> recipe for {rec['yield']['servings']} servings under {rec['time_total_min']} minutes."
    lines = [prompt, "###", rec["title"], "ingredients:"]
    for ing in rec["ingredients"]:
        lines.append(ingredient_line(ing, t["unit"]))
    lines.append("steps:")
    for i, s in enumerate(rec["steps"], 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)

def convert_dir(in_dir: str, out: str, fmt: str = "tagged", shards: int = 0,
                max_examples: int = 0, newline_blocks: bool = True):
    files = find_jsonl_files(in_dir)
    renderer = {"tagged": render_tagged, "plain": render_plain, "instruct": render_instruct}[fmt]

    def writer(path):
        if path.endswith(".gz"):
            return gzip.open(path, "wt", encoding="utf-8")
        return open(path, "w", encoding="utf-8")

    if shards and shards > 1 and not out.endswith(".txt") and not out.endswith(".txt.gz"):
        os.makedirs(out, exist_ok=True)
        shard_open = None
        shard_idx = 0
        written = 0
        per = None  # we'll do round-robin across shards

        handlers = []
        for i in range(shards):
            fn = f"corpus_{i:04d}.txt"
            p = os.path.join(out, fn)
            handlers.append(writer(p))

        try:
            for fp in files:
                with open_maybe_gzip(fp, "rt") as fh:
                    for line in fh:
                        if max_examples and written >= max_examples:
                            break
                        rec = json.loads(line)
                        txt = renderer(rec)
                        if newline_blocks:
                            txt += "\n\n"
                        # pick shard by modulo
                        h = handlers[written % shards]
                        h.write(txt)
                        written += 1
                if max_examples and written >= max_examples:
                    break
        finally:
            for h in handlers:
                h.close()
    else:
        # Single file
        out_dir = os.path.dirname(out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with writer(out) as oh:
            written = 0
            for fp in files:
                with open_maybe_gzip(fp, "rt") as fh:
                    for line in fh:
                        if max_examples and written >= max_examples:
                            break
                        rec = json.loads(line)
                        txt = renderer(rec)
                        oh.write(txt)
                        oh.write("\n\n" if newline_blocks else "\n")
                        written += 1
                if max_examples and written >= max_examples:
                    break
    return True

def main():
    ap = argparse.ArgumentParser(description="Convert recipelets JSONL to plain text")
    ap.add_argument("--in_dir", type=str, default='./recipelets', help="Directory containing recipelets_*.jsonl(.gz)")
    ap.add_argument("--out", type=str, default='./corpus_plain.txt', help="Output .txt (or directory if --shards>1)")
    ap.add_argument("--format", type=str, default='plain', choices=["tagged","plain","instruct"], help="Text format")
    ap.add_argument("--shards", type=int, default=0, help="If >1 and --out is a directory, write multiple text shards")
    ap.add_argument("--max_examples", type=int, default=0, help="If >0, cap number of converted examples")
    ap.add_argument("--no_blank_lines", action="store_true", help="Do not add blank line between samples")
    args = ap.parse_args()

    convert_dir(
        in_dir=args.in_dir,
        out=args.out,
        fmt=args.format,
        shards=args.shards,
        max_examples=args.max_examples,
        newline_blocks=not args.no_blank_lines
    )

if __name__ == "__main__":
    main()
