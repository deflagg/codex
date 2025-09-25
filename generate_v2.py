#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recipelets Generator (v2 - policy-hardened)
-------------------------------------------
This version bakes in strict data-hygiene rules so the generated corpus is
consistent for tokenization and tiny-model pretraining.

Decisions applied:
- No impossible references: steps only mention items that exist in ingredients.
- No "None" artifacts in steps.
- Avoid weird unit pairings (e.g., "eggs shrimp").
  * Only "egg" count-unit can be used for the "eggs" item.
  * "clove" count-unit only for "garlic".
  * Removed generic "piece" counts for proteins to avoid awkward phrasing.
- Quantities bounded to sane ranges; conversions to metric are correct and clamped.
- Unicode normalization: NFKC; dashes normalized to ASCII "-".
- Whitespace: no trailing spaces; text rendering uses one blank line between samples.
- Casing: all lower-case for titles and steps; headings are lower-case.
- Units & symbols standardized: tsp, tbsp, cup(s), g, ml, min.
  * tsp/tbsp never pluralized; cup is pluralized; g and ml have no plurals.
- Numbers: formatted with up to 2 decimals (strip trailing zeros). g/ml rendered as ints.

You can still produce JSONL (default). Optionally export a plain-text corpus with
"--text_out dataset.txt", which places exactly one blank line between samples.
"""

import os, json, random, uuid, math, gzip, argparse, hashlib, re, sys, time, unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# -----------------------------
# Formatting / Hygiene Policies
# -----------------------------
DECIMAL_PLACES = 2  # numeric precision for non-ml/g units
TEXT_LOWERCASE = True  # lower-case titles/steps
UNICODE_FORM = "NFKC"  # NFC or NFKC
NORMALIZE_DASHES = True  # replace en/em dashes with "-"

def nfkc(s: str) -> str:
    s = unicodedata.normalize(UNICODE_FORM, s)
    if NORMALIZE_DASHES:
        s = s.replace("–", "-").replace("—", "-")
    # collapse internal whitespace; keep newlines
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()

def fmt_num(q: float, unit: str) -> str:
    # g/ml rendered as integers
    if unit in ("g", "ml"):
        return str(int(round(q)))
    # otherwise fixed precision, trimmed
    s = f"{q:.{DECIMAL_PLACES}f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s

def unit_token(unit: str, qty: float) -> str:
    # tsp/tbsp never pluralized; g/ml never pluralized; cup pluralized; count-nouns pluralized
    if unit in ("g", "ml", "tsp", "tbsp"):
        return unit
    if unit == "cup":
        return "cup" if abs(qty - 1.0) < 1e-9 else "cups"
    irregular = {"clove": "cloves", "leaf": "leaves", "slice": "slices", "egg": "eggs"}
    if abs(qty - 1.0) < 1e-9:
        return unit
    return irregular.get(unit, unit + "s")

def to_lower(s: str) -> str:
    return s.lower() if TEXT_LOWERCASE else s

# -----------------------------
# Utilities
# -----------------------------
def slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

def tokens(s: str) -> List[str]:
    return [t for t in re.split(r"\s+", s.strip()) if t]

def choice_w(dist: Dict[str, float]) -> str:
    keys = list(dist.keys())
    weights = [dist[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]

def clamp(x, a, b):
    return max(a, min(b, x))

def rnd_step(n: float, step: float) -> float:
    return round(n / step) * step

# -----------------------------
# Ontology
# -----------------------------

CUISINES = ["generic","italian","mexican","indian","mediterranean","middle-eastern","east-asian"]
CUISINE_W = {
    "generic": 0.40,
    "italian": 0.12,
    "mexican": 0.12,
    "indian": 0.12,
    "mediterranean": 0.12,
    "middle-eastern": 0.06,
    "east-asian": 0.06,
}

DIETS = ["omni","veg","vegan","dairy-free","gluten-free"]
DIET_W = {"omni":0.55,"veg":0.25,"vegan":0.15,"dairy-free":0.03,"gluten-free":0.02}

METHODS = ["saute","boil","stir-fry","mix-only","roast"]
METHOD_W = {"saute":0.35,"boil":0.30,"stir-fry":0.20,"mix-only":0.10,"roast":0.05}

UNITS_US = ["tsp","tbsp","cup"]
UNITS_METRIC = ["g","ml"]
# Restrict count-units to the ones that don't cause awkwardness
COUNT_UNITS = ["clove","slice","leaf","egg"]

VERBS = {
    "heat": ["heat"],
    "add": ["add","stir in","tip in"],
    "cook": ["cook","sauté","fry","simmer","boil","toss"],
    "finish": ["finish with","stir in","fold in"],
    "serve": ["serve","plate","enjoy"],
}
HEAT_LEVELS = ["low","medium","medium-high","high"]
PANS = ["pan","skillet","pot","wok","saucepan"]

ING = {
    "aromatic": ["onion","garlic","scallion","ginger","shallot"],
    "herb": ["parsley","cilantro","basil","dill","thyme","rosemary","mint"],
    "spice": ["cumin","paprika","chili powder","turmeric","black pepper","oregano","coriander"],
    "fat": ["olive oil","butter","ghee","sesame oil"],
    "acid": ["lemon juice","lime juice","rice vinegar","white vinegar","balsamic vinegar"],
    "protein": ["chicken breast","tofu","chickpeas","eggs","shrimp"],
    "veg": ["tomato","spinach","bell pepper","broccoli","zucchini","mushroom","cucumber","carrot"],
    "carb": ["rice","pasta","quinoa","couscous","noodles","bread"],
    "dairy": ["yogurt","milk","cream","feta","parmesan","coconut milk"],
    "sweet": ["honey","sugar","maple syrup"],
    "liquid": ["water","vegetable broth","chicken broth","soy sauce"],
}
ITEM2CAT = {item:cat for cat, items in ING.items() for item in items}

CUISINE_CUES = {
    "generic": {"prefer": [], "avoid_pairs": []},
    "italian": {"prefer": ["basil","parmesan","olive oil","tomato","oregano","pasta"], "avoid_pairs": [("soy sauce","parmesan"),("soy sauce","basil"),("ginger","parmesan")]},
    "mexican": {"prefer": ["cilantro","lime juice","cumin","chili powder","black pepper","tomato"], "avoid_pairs": [("soy sauce","cilantro")]},
    "indian": {"prefer": ["turmeric","coriander","cumin","ghee","ginger","garlic"], "avoid_pairs": [("parmesan","turmeric")]},
    "mediterranean": {"prefer": ["olive oil","lemon juice","parsley","feta","tomato","oregano"], "avoid_pairs": [("soy sauce","feta")]},
    "middle-eastern": {"prefer": ["cumin","parsley","lemon juice","yogurt","coriander","olive oil"], "avoid_pairs": [("soy sauce","yogurt")]},
    "east-asian": {"prefer": ["soy sauce","ginger","sesame oil","scallion","noodles"], "avoid_pairs": [("parmesan","soy sauce"),("parmesan","ginger")]},
}

NON_VEGAN = set(["chicken breast","shrimp","eggs","yogurt","milk","cream","feta","parmesan","honey","chicken broth","ghee"])
NON_VEGETARIAN = set(["chicken breast","shrimp","chicken broth"])
DAIRY_ITEMS = set(["yogurt","milk","cream","feta","parmesan","ghee"])
GLUTEN_ITEMS = set(["pasta","bread","couscous"])

# Unit compatibility per category
UNIT_COMPAT = {
    "aromatic": {"us":["tsp","tbsp"], "metric":["g"], "count":["clove"]},  # (clove restricted to garlic later)
    "herb": {"us":["tsp","tbsp"], "metric":["g"], "count":["leaf"]},
    "spice": {"us":["tsp","tbsp"], "metric":["g"], "count":[]},
    "fat": {"us":["tsp","tbsp","cup"], "metric":["ml","g"], "count":[]},
    "acid": {"us":["tsp","tbsp"], "metric":["ml"], "count":[]},
    "protein": {"us":["cup"], "metric":["g"], "count":[]},  # no generic count units for proteins
    "veg": {"us":["cup"], "metric":["g"], "count":["slice"]},
    "carb": {"us":["cup"], "metric":["g"], "count":[]},
    "dairy": {"us":["tbsp","cup"], "metric":["g","ml"], "count":[]},
    "sweet": {"us":["tsp","tbsp"], "metric":["g","ml"], "count":[]},
    "liquid": {"us":["cup","tbsp"], "metric":["ml"], "count":[]},
}

# Quantity ranges by unit (tight bounds to avoid absurd magnitudes)
QTY_RANGE = {
    "tsp": (0.25, 3.0, 0.25),
    "tbsp": (0.5, 4.0, 0.5),
    "cup": (0.25, 2.0, 0.25),
    "g": (5, 500, 5),
    "ml": (5, 500, 5),
    "clove": (1, 3, 1),
    "slice": (1, 6, 1),
    "leaf": (2, 12, 1),
    "egg": (1, 4, 1),
}

TITLE_HEAD_ADJ = ["quick","easy","simple","spicy","zesty","garlic","herb","lemon","fresh","fast"]

# -----------------------------
# Distributions
# -----------------------------
UNIT_SYSTEM_W = {"us":0.5, "metric":0.5}
LEN_W = {"short":0.7, "med":0.3}
LEVEL_W = {1:0.35, 2:0.35, 3:0.2, 4:0.07, 5:0.03}
METHOD_TO_STEP_BOUNDS = {
    "mix-only": (3,3),
    "saute": (3,5),
    "stir-fry": (3,5),
    "boil": (3,4),
    "roast": (4,5),
}
ING_COUNT_W = {3:0.15, 4:0.35, 5:0.35, 6:0.15}
SERVINGS_W = {2:0.55,3:0.25,4:0.20}
TIME_BUCKETS = [5,10,15,20,30]
TIME_W = {5:0.05,10:0.40,15:0.35,20:0.20,30:0.05}

# -----------------------------
# Constraint helpers
# -----------------------------

def violates_cuisine_pairs(cuisine: str, items: List[str]) -> bool:
    avoid = CUISINE_CUES.get(cuisine, {}).get("avoid_pairs", [])
    s = set(items)
    for a,b in avoid:
        if a in s and b in s:
            return True
    return False

def diet_ok(diet: str, items: List[str]) -> bool:
    s = set(items)
    if diet == "vegan" and (s & NON_VEGAN):
        return False
    if diet == "veg" and (s & NON_VEGETARIAN):
        return False
    if diet == "dairy-free" and (s & DAIRY_ITEMS):
        return False
    if diet == "gluten-free" and (s & GLUTEN_ITEMS):
        return False
    return True

def method_requirements(method: str) -> List[str]:
    if method == "saute":
        return ["fat","aromatic"]
    if method == "stir-fry":
        return ["fat","aromatic","veg"]
    if method == "boil":
        return ["liquid"]
    if method == "mix-only":
        return ["fat","acid","herb"]
    if method == "roast":
        return ["fat","veg"]
    return []

def method_forbids(method: str, items: List[str]) -> bool:
    if method == "mix-only":
        hot_only = {"chicken breast","shrimp","eggs","pasta","rice","quinoa","noodles","broccoli"}
        if set(items) & hot_only:
            return True
    if method == "roast":
        if all(ITEM2CAT[i] in ("liquid","acid","sweet","spice","herb","fat") for i in items):
            return True
    return False

# Item-specific unit overrides / filters
def allowed_units_for_item(item: str, unit_system: str) -> List[str]:
    cat = ITEM2CAT[item]
    compat = UNIT_COMPAT[cat]
    pool = []
    if unit_system == "us":
        pool.extend(compat.get("us", []))
    else:
        pool.extend(compat.get("metric", []))
    pool.extend(compat.get("count", []))

    # Restrict awkward count-nouns
    if "egg" in pool and item != "eggs":
        pool.remove("egg")
    if "clove" in pool and item != "garlic":
        pool.remove("clove")

    # sanity: unique
    return list(dict.fromkeys(pool))

def sample_unit_for_item(item: str, unit_system: str) -> str:
    pool = allowed_units_for_item(item, unit_system)
    if not pool:
        # safety fallback per category
        cat = ITEM2CAT[item]
        return "g" if unit_system == "metric" else ("tbsp" if cat in ("fat","spice","herb","sweet","acid","aromatic") else "cup")
    return random.choice(pool)

def sample_qty(unit: str) -> float:
    lo, hi, step = QTY_RANGE[unit]
    if isinstance(lo, int) and isinstance(hi, int) and step == 1:
        return random.randint(lo, hi)
    val = random.uniform(lo, hi)
    return round(val / step) * step

# -----------------------------
# Text helpers
# -----------------------------

def step_trim(s: str, max_tokens: int = 12) -> str:
    toks = tokens(s)
    if len(toks) <= max_tokens:
        return s
    return " ".join(toks[:max_tokens]).rstrip(",. ") + "."

def normalize_step_text(s: str) -> str:
    s = nfkc(s)
    # standardize minute units
    s = re.sub(r"\bmins?\b", "min", s)
    # lower-case policy
    s = to_lower(s)
    # ensure sentence ends with a period
    s = s.rstrip()
    if not s.endswith("."):
        s += "."
    return s

def normalize_title(t: str) -> str:
    t = nfkc(t)
    t = re.sub(r"\s+", " ", t).strip()
    return to_lower(t)

# -----------------------------
# Title & Steps
# -----------------------------

def title_from(method: str, items: List[str], cuisine: str) -> str:
    def pick_head():
        pri = [i for i in items if ITEM2CAT[i] in ("protein","veg","carb")]
        return random.choice(pri) if pri else random.choice(items)
    head = pick_head()
    head_noun = head.split()[-1] if " " in head else head
    adj = random.choice(TITLE_HEAD_ADJ)
    mtxt = {"saute":"saute", "stir-fry":"stir-fry"}.get(method, method)
    parts = []
    if random.random() < 0.6:
        parts.append(adj)
    if random.random() < 0.6:
        cues = CUISINE_CUES.get(cuisine,{}).get("prefer",[])
        cue = random.choice(cues) if cues else None
        if cue and cue not in head:
            parts.append(cue.split()[-1])
    parts.extend([head_noun, mtxt])
    title = " ".join(parts[:6])
    return normalize_title(title)

def build_steps(method: str, ing_list: List[Dict], cuisine: str) -> List[str]:
    def find_item(cat):
        for ing in ing_list:
            if ing["category"] == cat:
                return ing["item"]
        return None
    fat = find_item("fat")            # must exist for saute/stir-fry/roast by constraints
    aromatic = find_item("aromatic")  # must exist for saute/stir-fry by constraints
    liquid = find_item("liquid")      # must exist for boil by constraints
    acid = find_item("acid")
    herb = find_item("herb")
    protein = find_item("protein")
    veg = find_item("veg")
    carb = find_item("carb")

    steps = []
    if method in ("saute","stir-fry"):
        steps.append(f"{random.choice(VERBS['heat'])} {fat} in a {random.choice(PANS)} over {random.choice(HEAT_LEVELS)} heat.")
        if aromatic:
            steps.append(f"{random.choice(VERBS['add'])} {aromatic}; cook 2-3 min until fragrant.")
        core = protein or veg or carb
        if core:
            steps.append(f"{random.choice(VERBS['add'])} {core}; {random.choice(VERBS['cook'])} 4-6 min.")
        fin = acid or herb
        if fin:
            steps.append(f"{random.choice(VERBS['finish'])} {fin}; {random.choice(VERBS['cook'])} 1-2 min.")
        steps.append(f"{random.choice(VERBS['serve'])} warm.")
    elif method == "boil":
        steps.append(f"bring {liquid} to a boil.")
        core = carb or veg or protein
        steps.append(f"{random.choice(VERBS['add'])} {core}; {random.choice(VERBS['cook'])} until tender.")
        if herb or acid or fat:
            fin = herb or acid or fat
            steps.append(f"{random.choice(VERBS['finish'])} {fin}; toss to coat.")
        steps.append(f"{random.choice(VERBS['serve'])} hot.")
    elif method == "mix-only":
        raw_ok = {"tomato","cucumber","feta","chickpeas","tofu","spinach"}
        base = None
        for ing in ing_list:
            if ing["item"] in raw_ok or ing["category"] in ("veg","herb","dairy"):
                base = ing["item"]
                break
        if base is None:
            base = veg or herb or (protein if protein in ("tofu","chickpeas") else None)
        # base MUST come from ingredients; if not found, fall back to the first non-fat/acid/liquid
        if base is None:
            for ing in ing_list:
                if ing["category"] not in ("fat","acid","liquid"):
                    base = ing["item"]
                    break
        steps.append(f"combine {base} with {fat}.")
        if acid:
            steps.append(f"{random.choice(VERBS['finish'])} {acid} and {herb or 'spice'}; mix well.")
        else:
            steps.append(f"add {herb or 'spice'}; mix well.")
        steps.append(f"season to taste; {random.choice(VERBS['serve'])} right away.")
    elif method == "roast":
        steps.append(f"toss vegetables with {fat}.")
        if aromatic:
            steps.append(f"{random.choice(VERBS['add'])} {aromatic}; mix to coat.")
        steps.append("roast at 220°c / 425°f for 15-20 min.")
        if herb or acid:
            steps.append(f"{random.choice(VERBS['finish'])} {herb or acid}.")
        steps.append(f"{random.choice(VERBS['serve'])}.")
    else:
        steps = ["combine ingredients.","cook briefly.","season and serve."]

    # trim and normalize
    steps = [normalize_step_text(step_trim(s)) for s in steps]

    # enforce method-specific step count bounds
    lo, hi = METHOD_TO_STEP_BOUNDS.get(method, (3,5))
    if len(steps) < lo:
        while len(steps) < lo:
            steps.append("season and serve.")
    if len(steps) > hi:
        steps = steps[:hi]
    return steps

def servings_sample() -> int:
    return int(choice_w(SERVINGS_W))

def time_bucket_sample() -> int:
    return int(choice_w(TIME_W))

def len_tag_sample() -> str:
    return choice_w(LEN_W)

def level_sample() -> int:
    return int(choice_w(LEVEL_W))

def unit_system_sample() -> str:
    return choice_w(UNIT_SYSTEM_W)

def ing_count_sample() -> int:
    return int(choice_w(ING_COUNT_W))

# -----------------------------
# Ingredient selection
# -----------------------------

def pick_ingredients(method: str, cuisine: str, diet: str, unit_system: str, desired_count: int) -> Optional[List[Dict]]:
    req_cats = method_requirements(method)
    all_items = [i for cat, items in ING.items() for i in items]

    def diet_filter(i):
        if diet == "vegan" and i in NON_VEGAN: return False
        if diet == "veg" and i in NON_VEGETARIAN: return False
        if diet == "dairy-free" and i in DAIRY_ITEMS: return False
        if diet == "gluten-free" and i in GLUTEN_ITEMS: return False
        return True

    pool = [i for i in all_items if diet_filter(i)]

    prefer = set(CUISINE_CUES.get(cuisine,{}).get("prefer",[]))
    weighted = []
    for i in pool:
        w = 1.0 + (0.7 if i in prefer else 0.0)
        weighted.extend([i]*int(w*10))

    chosen = []
    chosen_cats = set()
    for cat in req_cats:
        options = [i for i in weighted if ITEM2CAT[i] == cat]
        if not options:
            return None
        item = random.choice(options)
        chosen.append(item)
        chosen_cats.add(ITEM2CAT[item])

    while len(chosen) < desired_count:
        cat_options = [c for c in ING.keys() if sum(1 for x in chosen if ITEM2CAT[x]==c) < 2]
        if not cat_options:
            break
        cat = random.choice(cat_options)
        options = [i for i in weighted if ITEM2CAT[i]==cat and i not in chosen]
        if not options:
            options = [i for i in weighted if i not in chosen]
            if not options:
                break
        item = random.choice(options)
        chosen.append(item)
        if violates_cuisine_pairs(cuisine, chosen):
            chosen.pop()
            continue

    if len(chosen) < max(3, len(req_cats)):
        return None
    if method_forbids(method, chosen):
        return None

    ing_list = []
    for it in chosen:
        unit = sample_unit_for_item(it, unit_system)
        qty = sample_qty(unit)
        ing_list.append({
            "item": it,
            "category": ITEM2CAT[it],
            "qty": qty,
            "unit": unit
        })
    return ing_list

# -----------------------------
# Record assembly & validation
# -----------------------------

def canonical_key(rec: Dict) -> str:
    cats = sorted([i["category"] for i in rec["ingredients"]])
    head = rec["title"].split()[:2]
    return "|".join([rec["tags"]["cuisine"], rec["tags"]["method"], " ".join(head), ",".join(cats), rec["tags"]["unit"], rec["tags"]["diet"]])

def ingredient_line(ing: Dict, unit_system: str) -> str:
    qty = ing["qty"]
    orig_unit = ing["unit"]
    unit = orig_unit

    # Convert US -> metric if needed (correctly) and clamp ranges
    if unit_system == "metric" and orig_unit in ("cup","tbsp","tsp"):
        conv_ml = {"cup": 240, "tbsp": 15, "tsp": 5}[orig_unit]
        unit = "ml"
        qty = qty * conv_ml
        lo, hi, _ = QTY_RANGE["ml"]
        qty = clamp(qty, lo, hi)

    qty_txt = fmt_num(qty, unit)
    unit_txt = unit_token(unit, qty)
    return f"{qty_txt} {unit_txt} {ing['item']}"

def record_text_view(rec: Dict) -> str:
    tags = rec["tags"]
    head = f"<unit={tags['unit']}> <cuisine={tags['cuisine']}> <diet={tags['diet']}> <lvl={tags['lvl']}>"
    lines = [head, rec["title"], "ingredients:"]
    for ing in rec["ingredients"]:
        lines.append("- " + ingredient_line(ing, tags["unit"]))
    lines.append("steps:")
    for idx, s in enumerate(rec["steps"], 1):
        lines.append(f"{idx}. {s}")
    # normalize unicode, dashes, whitespace; ensure no trailing spaces
    text = "\n".join(lines)
    text = nfkc(text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return text

def validate_record(rec: Dict) -> Tuple[bool, List[str]]:
    issues = []
    tags = rec["tags"]
    items = [i["item"] for i in rec["ingredients"]]
    if not diet_ok(tags["diet"], items):
        issues.append("diet_violation")
    if method_forbids(tags["method"], items):
        issues.append("method_violation")
    if tags["unit"] == "metric":
        txt = record_text_view(rec)
        if re.search(r"\b(cup|tbsp|tsp)\b", txt):
            issues.append("unit_system_violation")
    for s in rec["steps"]:
        if len(tokens(s)) > 14:
            issues.append("step_length")
            break
    if not (3 <= len(rec["ingredients"]) <= 6):
        issues.append("ingredient_count")
    lb, hb = METHOD_TO_STEP_BOUNDS.get(tags["method"], (3,5))
    if not (lb <= len(rec["steps"]) <= hb):
        issues.append("step_count")
    if violates_cuisine_pairs(tags["cuisine"], items):
        issues.append("cuisine_pair")
    # impossible reference check: any words that look like ingredients but not present
    joined = " ".join(rec["steps"])
    mentioned = set()
    for item in ITEM2CAT.keys():
        if re.search(rf"\b{re.escape(item)}\b", joined):
            mentioned.add(item)
    if not mentioned.issubset(set(items)):
        # if we intentionally referenced something not in ingredients, mark it
        issues.append("step_refers_missing_ingredient")
    return (len(issues)==0, issues)

def sample_tags() -> Dict:
    return {
        "unit": "metric" if random.random()<0.5 else "us",
        "cuisine": choice_w(CUISINE_W),
        "diet": choice_w(DIET_W),
        "lvl": level_sample(),
        "len": len_tag_sample(),
        "method": choice_w(METHOD_W),
        "style": random.choice(["neutral","minimal","cheerful"]),
    }

def assemble_record(rng_seed: Optional[int]=None) -> Optional[Dict]:
    tags = sample_tags()
    desired_ing = ing_count_sample()
    if tags["lvl"] >= 4 and desired_ing < 4:
        desired_ing = 4
    ing_list = pick_ingredients(tags["method"], tags["cuisine"], tags["diet"], tags["unit"], desired_ing)
    if not ing_list:
        return None
    steps = build_steps(tags["method"], ing_list, tags["cuisine"])
    items = [i["item"] for i in ing_list]
    title = title_from(tags["method"], items, tags["cuisine"])
    rec = {
        "id": uuid.uuid4().hex,
        "tags": tags,
        "title": title,
        "yield": {"servings": servings_sample()},
        "time_total_min": time_bucket_sample(),
        "ingredients": ing_list,
        "steps": steps,
    }

    ok, issues = validate_record(rec)
    if not ok:
        # If the ONLY issue is "step_refers_missing_ingredient" due to weird edge,
        # try to repair by adding the first missing mentioned ingredient with a minimal qty.
        if issues == ["step_refers_missing_ingredient"]:
            joined = " ".join(rec["steps"])
            for item in ITEM2CAT.keys():
                if item in items:
                    continue
                if re.search(rf"\b{re.escape(item)}\b", joined):
                    # add with a sane, small quantity
                    unit = sample_unit_for_item(item, rec["tags"]["unit"])
                    qty = sample_qty(unit)
                    rec["ingredients"].append({"item": item, "category": ITEM2CAT[item], "qty": qty, "unit": unit})
                    break
            ok2, issues2 = validate_record(rec)
            if ok2:
                return rec
        return None
    return rec

# -----------------------------
# Generation loop
# -----------------------------

def generate_dataset(total: int, out_dir: str, shards: int = 100, gzip_out: bool = False,
                     seed: int = 42, noise_rate: float = 0.05, log_every: int = 5000,
                     text_out: Optional[str] = None):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    per_shard = max(1, total // shards)
    shards = max(1, min(shards, total))

    t0 = time.time()
    written = 0
    attempts = 0

    def writer_handle(idx: int):
        fn = f"recipelets_{idx:04d}.jsonl"
        path = os.path.join(out_dir, fn + (".gz" if gzip_out else ""))
        if gzip_out:
            return path, gzip.open(path, "wt", encoding="utf-8")
        else:
            return path, open(path, "w", encoding="utf-8")

    text_fh = open(text_out, "w", encoding="utf-8") if text_out else None

    shard_idx = 0
    path, fh = writer_handle(shard_idx)

    try:
        while written < total:
            attempts += 1
            rec = assemble_record()
            if not rec:
                continue

            # Optional minimal noise (kept safe: no impossible references introduced)
            if noise_rate > 0 and random.random() < noise_rate:
                rec["time_total_min"] = max(5, min(30, rec["time_total_min"] + random.choice([-5, 5])))
                rec["id"] = uuid.uuid4().hex

            ok, issues = validate_record(rec)
            if not ok:
                if noise_rate > 0 and len(issues) <= 2 and "step_refers_missing_ingredient" not in issues and random.random() < 0.5:
                    rec.setdefault("violations", issues)
                else:
                    continue

            key = canonical_key(rec)
            if key in seen:
                continue
            seen.add(key)

            line = json.dumps(rec, ensure_ascii=False)
            fh.write(line + "\n")
            if text_fh:
                text_fh.write(record_text_view(rec) + "\n\n")

            written += 1

            if written % per_shard == 0 and written < total:
                fh.close()
                shard_idx += 1
                path, fh = writer_handle(shard_idx)

            if written % log_every == 0:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                print(f"[progress] {written}/{total} written | {rate:.1f} rec/s | shards={shard_idx+1}", flush=True)

    finally:
        fh.close()
        if text_fh:
            text_fh.close()

    manifest = {
        "total": written,
        "shards": shard_idx + 1,
        "per_shard": per_shard,
        "seed": seed,
        "gzip": gzip_out,
        "noise_rate": noise_rate,
        "text_out": text_out,
        "policies": {
            "unicode_form": UNICODE_FORM,
            "dash_normalization": NORMALIZE_DASHES,
            "lowercase": TEXT_LOWERCASE,
            "decimal_places": DECIMAL_PLACES,
        }
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"Done. Wrote {written} records across {shard_idx+1} shard(s) to {out_dir}")

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic recipelets dataset (policy-hardened)")
    ap.add_argument("--total", type=int, default=100000, help="Total number of records to generate")
    ap.add_argument("--out_dir", type=str, default="./recipelets", help="Output directory")
    ap.add_argument("--shards", type=int, default=1, help="Number of shards (files)")
    ap.add_argument("--seed", type=int, default=43, help="RNG seed")
    ap.add_argument("--gzip", action="store_true", help="Write .jsonl.gz shards")
    ap.add_argument("--noise_rate", type=float, default=0.05, help="Rate of light noise injection")
    ap.add_argument("--text_out", type=str, default=None, help="Optional path to also write a plain-text corpus with 1 blank line between samples")
    args = ap.parse_args()

    generate_dataset(total=args.total, out_dir=args.out_dir, shards=args.shards,
                     gzip_out=args.gzip, seed=args.seed, noise_rate=args.noise_rate,
                     text_out=args.text_out)

if __name__ == "__main__":
    main()
