#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recipelets Generator
--------------------
Generate up to ~1,000,000 short, structured "recipelets" suitable for tiny decoder-only models.

Key features:
- Tight, controllable ontology (ingredients, units, cuisines, methods)
- Short sequences (titles ≤ ~6 tokens; 3–5 steps ≤ ~12 tokens each)
- Diet & method & unit constraints
- Optional light noise injection for robustness
- Sharded JSONL output with simple de-duplication

Usage (examples):
    python recipelets_generator.py --total 1000000 --out_dir ./recipelets --shards 1000 --gzip
    python recipelets_generator.py --total 20000 --out_dir ./demo --shards 20 --seed 7

Author: ChatGPT (GPT-5 Pro)
"""

import os, json, random, uuid, math, gzip, argparse, itertools, hashlib, re, sys, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# -----------------------------
# Utilities
# -----------------------------

def slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

def tokens(s: str) -> List[str]:
    # Very simple tokenization; good enough for length budgeting
    return [t for t in re.split(r"\s+", s.strip()) if t]

def choice_w(dist: Dict[str, float]) -> str:
    keys = list(dist.keys())
    weights = [dist[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]

def clamp(x, a, b):
    return max(a, min(b, x))

def rnd_step(n: float, step: float) -> float:
    return round(n / step) * step

def pluralize(unit: str, qty: float) -> str:
    # simple pluralization for units
    if unit in ("g","ml"):
        return unit  # no plural form
    if abs(qty - 1.0) < 1e-9:
        return unit
    # irregulars for counts
    irregular = {"clove":"cloves","leaf":"leaves","slice":"slices","piece":"pieces","egg":"eggs"}
    if unit in irregular:
        return irregular[unit]
    return unit + "s"

def a_or_an(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"

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
COUNT_UNITS = ["clove","slice","leaf","piece","egg"]  # can appear in either unit system

# verbs & concise phrases to keep steps short (≤ ~12 tokens per step)
VERBS = {
    "heat": ["Heat"],
    "add": ["Add","Stir in","Tip in"],
    "cook": ["Cook","Sauté","Fry","Simmer","Boil","Toss"],
    "finish": ["Finish with","Stir in","Fold in"],
    "serve": ["Serve","Plate","Enjoy"],
}
HEAT_LEVELS = ["low","medium","medium-high","high"]
PANS = ["pan","skillet","pot","wok","saucepan"]

# Ingredient ontology by category
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

# Fast reverse map
ITEM2CAT = {item:cat for cat, items in ING.items() for item in items}

# Cuisine cues (positive lists) and incompatibilities (negative pairs)
CUISINE_CUES = {
    "generic": {"prefer": [], "avoid_pairs": []},
    "italian": {"prefer": ["basil","parmesan","olive oil","tomato","oregano","pasta"], "avoid_pairs": [("soy sauce","parmesan"),("soy sauce","basil"),("ginger","parmesan")]},
    "mexican": {"prefer": ["cilantro","lime juice","cumin","chili powder","black pepper","tomato"], "avoid_pairs": [("soy sauce","cilantro")]},
    "indian": {"prefer": ["turmeric","coriander","cumin","ghee","ginger","garlic"], "avoid_pairs": [("parmesan","turmeric")]},
    "mediterranean": {"prefer": ["olive oil","lemon juice","parsley","feta","tomato","oregano"], "avoid_pairs": [("soy sauce","feta")]},
    "middle-eastern": {"prefer": ["cumin","parsley","lemon juice","yogurt","coriander","olive oil"], "avoid_pairs": [("soy sauce","yogurt")]},
    "east-asian": {"prefer": ["soy sauce","ginger","sesame oil","scallion","noodles"], "avoid_pairs": [("parmesan","soy sauce"),("parmesan","ginger")]},
}

# Diet exclusions
NON_VEGAN = set(["chicken breast","shrimp","eggs","yogurt","milk","cream","feta","parmesan","honey","chicken broth","ghee"])
NON_VEGETARIAN = set(["chicken breast","shrimp","chicken broth"])  # eggs/dairy allowed for veg
DAIRY_ITEMS = set(["yogurt","milk","cream","feta","parmesan","ghee"])
GLUTEN_ITEMS = set(["pasta","bread","couscous"])  # treat couscous as gluten for simplicity

# Unit compatibility per category
UNIT_COMPAT = {
    "aromatic": {"us":["tsp","tbsp"], "metric":["g"], "count":["clove"]},  # garlic often in cloves
    "herb": {"us":["tsp","tbsp"], "metric":["g"], "count":["leaf"]},
    "spice": {"us":["tsp","tbsp"], "metric":["g"], "count":[]},
    "fat": {"us":["tsp","tbsp"], "metric":["ml","g"], "count":[]},
    "acid": {"us":["tsp","tbsp"], "metric":["ml"], "count":[]},
    "protein": {"us":["cup"], "metric":["g"], "count":["piece","egg"]},
    "veg": {"us":["cup"], "metric":["g"], "count":["slice","piece"]},
    "carb": {"us":["cup"], "metric":["g"], "count":["slice","piece"]},
    "dairy": {"us":["tbsp","cup"], "metric":["g","ml"], "count":[]},
    "sweet": {"us":["tsp","tbsp"], "metric":["g","ml"], "count":[]},
    "liquid": {"us":["cup","tbsp"], "metric":["ml"], "count":[]},
}

# Quantity ranges by unit
QTY_RANGE = {
    "tsp": (0.25, 3.0, 0.25),
    "tbsp": (0.5, 4.0, 0.5),
    "cup": (0.25, 2.0, 0.25),
    "g": (5, 500, 5),
    "ml": (5, 500, 5),
    "clove": (1, 3, 1),
    "slice": (1, 6, 1),
    "leaf": (2, 12, 1),
    "piece": (1, 4, 1),
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
    if diet == "vegan":
        if s & NON_VEGAN:
            return False
    if diet == "veg":
        if s & NON_VEGETARIAN:
            return False
    if diet == "dairy-free":
        if s & DAIRY_ITEMS:
            return False
    if diet == "gluten-free":
        if s & GLUTEN_ITEMS:
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
        # forbid items that scream "needs cooking"? okay to keep protein/veg if raw-ish (cucumber, tomato, feta, chickpeas, tofu)
        hot_only = {"chicken breast","shrimp","eggs","pasta","rice","quinoa","noodles","broccoli"}
        if set(items) & hot_only:
            return True
    if method == "roast":
        # if liquid-only makes no sense
        if all(ITEM2CAT[i] in ("liquid","acid","sweet","spice","herb","fat") for i in items):
            return True
    return False

def sample_unit_for_item(item: str, unit_system: str) -> str:
    cat = ITEM2CAT[item]
    compat = UNIT_COMPAT[cat]
    pool = []
    if unit_system == "us":
        pool.extend(compat.get("us", []))
    else:
        pool.extend(compat.get("metric", []))
    pool.extend(compat.get("count", []))
    # avoid using count + non-count simultaneously later
    return random.choice(pool)

def sample_qty(unit: str) -> float:
    lo, hi, step = QTY_RANGE[unit]
    if isinstance(lo, int) and isinstance(hi, int) and step == 1:
        return random.randint(lo, hi)
    # round to the nearest step
    val = random.uniform(lo, hi)
    return round(val / step) * step

def title_from(method: str, items: List[str], cuisine: str) -> str:
    # Pick a head ingredient (prefer protein/veg/carb), and maybe a cue adj
    def pick_head():
        pri = [i for i in items if ITEM2CAT[i] in ("protein","veg","carb")]
        return random.choice(pri) if pri else random.choice(items)
    head = pick_head()
    head_noun = head.split()[-1] if " " in head else head
    adj = random.choice(TITLE_HEAD_ADJ)
    mtxt = {"saute":"sauté", "stir-fry":"stir-fry"}.get(method, method)
    parts = []
    if random.random() < 0.6:
        parts.append(adj)
    if random.random() < 0.6:
        # add a flavor cue if present
        cues = CUISINE_CUES.get(cuisine,{}).get("prefer",[])
        cue = random.choice(cues) if cues else None
        if cue and cue not in head:
            parts.append(cue.split()[-1])
    parts.extend([head_noun, mtxt])
    # clean and cap tokens to ~6
    title = " ".join(parts[:6]).lower()
    title = title.replace("  "," ").strip()
    return title

def step_trim(s: str, max_tokens: int = 12) -> str:
    toks = tokens(s)
    if len(toks) <= max_tokens:
        return s
    return " ".join(toks[:max_tokens]).rstrip(",. ") + "."

def build_steps(method: str, ing_list: List[Dict], cuisine: str) -> List[str]:
    # Extract some role players for template fills
    # Find one fat, aromatic, acid, liquid if present
    def find_item(cat):
        for ing in ing_list:
            if ing["category"] == cat:
                return ing["item"]
        return None
    fat = find_item("fat") or "oil"
    aromatic = find_item("aromatic")
    liquid = find_item("liquid") or ("water" if method in ("boil","roast") else None)
    acid = find_item("acid")
    herb = find_item("herb")
    protein = find_item("protein")
    veg = find_item("veg")
    carb = find_item("carb")

    steps = []
    if method in ("saute","stir-fry"):
        steps.append(f"{random.choice(VERBS['heat'])} {fat} in a {random.choice(PANS)} over {random.choice(HEAT_LEVELS)} heat.")
        if aromatic:
            steps.append(f"{random.choice(VERBS['add'])} {aromatic}; cook 2–3 min until fragrant.")
        core = protein or veg or carb
        if core:
            steps.append(f"{random.choice(VERBS['add'])} {core}; {random.choice(VERBS['cook'])} 4–6 min.")
        fin = acid or herb or liquid
        if fin:
            steps.append(f"{random.choice(VERBS['finish'])} {fin}; {random.choice(VERBS['cook'])} 1–2 min.")
        steps.append(f"{random.choice(VERBS['serve'])} warm.")
    elif method == "boil":
        steps.append(f"Bring {liquid or 'water'} to a boil.")
        core = carb or veg or protein
        steps.append(f"{random.choice(VERBS['add'])} {core}; {random.choice(VERBS['cook'])} until tender.")
        if herb or acid or fat:
            fin = herb or acid or fat
            steps.append(f"{random.choice(VERBS['finish'])} {fin}; toss to coat.")
        steps.append(f"{random.choice(VERBS['serve'])} hot.")
    elif method == "mix-only":
        base = veg or (protein if (protein in ('tofu','chickpeas')) else None) or herb or "tomato"
        steps.append(f"Combine {base} with {fat}.")
        if acid:
            steps.append(f"{random.choice(VERBS['finish'])} {acid} and {herb or 'spice'}; mix well.")
        else:
            steps.append(f"Add {herb or 'spice'}; mix well.")
        steps.append(f"Season to taste; {random.choice(VERBS['serve']).lower()} right away.")
    elif method == "roast":
        steps.append(f"Toss veg with {fat}.")
        if aromatic:
            steps.append(f"{random.choice(VERBS['add'])} {aromatic}; mix to coat.")
        steps.append("Roast at 220°C / 425°F for 15–20 min.")
        if herb or acid:
            steps.append(f"{random.choice(VERBS['finish'])} {herb or acid}.")
        steps.append(f"{random.choice(VERBS['serve'])}.")
    else:
        # fallback: generic concise steps
        steps = ["Combine ingredients.","Cook briefly.","Season and serve."]

    # Trim to keep steps concise
    steps = [step_trim(s) for s in steps]
    # Enforce method-specific step count bounds
    lo, hi = METHOD_TO_STEP_BOUNDS.get(method, (3,5))
    if len(steps) < lo:
        while len(steps) < lo:
            steps.append("Season and serve.")
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
    # Ensure requirements
    req_cats = method_requirements(method)
    # pool of allowed items respecting diet and cuisine synergy
    all_items = [i for cat, items in ING.items() for i in items]
    # Filter by diet first
    def diet_filter(i):
        if diet == "vegan" and i in NON_VEGAN: return False
        if diet == "veg" and i in NON_VEGETARIAN: return False
        if diet == "dairy-free" and i in DAIRY_ITEMS: return False
        if diet == "gluten-free" and i in GLUTEN_ITEMS: return False
        return True
    pool = [i for i in all_items if diet_filter(i)]

    # Bias pool by cuisine cues (soft preference)
    prefer = set(CUISINE_CUES.get(cuisine,{}).get("prefer",[]))
    weighted = []
    for i in pool:
        w = 1.0 + (0.7 if i in prefer else 0.0)
        weighted.extend([i]*int(w*10))

    # Start with required categories (choose one item per required category)
    chosen = []
    chosen_cats = set()
    for cat in req_cats:
        options = [i for i in weighted if ITEM2CAT[i] == cat]
        if not options:
            return None  # cannot satisfy method under current diet
        item = random.choice(options)
        chosen.append(item)
        chosen_cats.add(ITEM2CAT[item])

    # Fill up to desired_count with diverse categories
    while len(chosen) < desired_count:
        # Avoid excessive duplicates of same category (max 2 per cat)
        cat_options = [c for c in ING.keys() if sum(1 for x in chosen if ITEM2CAT[x]==c) < 2]
        if not cat_options:
            break
        cat = random.choice(cat_options)
        options = [i for i in weighted if ITEM2CAT[i]==cat and i not in chosen]
        if not options:
            # try any from pool
            options = [i for i in weighted if i not in chosen]
            if not options:
                break
        item = random.choice(options)
        chosen.append(item)

        # Avoid cuisine-prohibited pairs
        if violates_cuisine_pairs(cuisine, chosen):
            chosen.pop()
            continue

    # Basic sanity: check counts
    if len(chosen) < max(3, len(req_cats)):
        return None

    # Method-specific forbid check
    if method_forbids(method, chosen):
        return None

    # Build ingredient dicts with qty+unit
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

def normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def normalize_step(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def canonical_key(rec: Dict) -> str:
    # key balancing method, cuisine, sorted category multiset, and title head
    cats = sorted([i["category"] for i in rec["ingredients"]])
    head = rec["title"].split()[:2]
    return "|".join([rec["tags"]["cuisine"], rec["tags"]["method"], " ".join(head), ",".join(cats), rec["tags"]["unit"], rec["tags"]["diet"]])

def ingredient_line(ing: Dict, unit_system: str) -> str:
    qty = ing["qty"]
    unit = ing["unit"]
    # unit pluralization
    unit_txt = pluralize(unit, qty)
    # render qty compactly (avoid trailing .0 when integer-like)
    if isinstance(qty, float) and abs(qty - int(qty)) < 1e-9:
        qty_txt = str(int(qty))
    else:
        qty_txt = str(qty)
    # For metric-only constraint: disallow cup/tsp/tbsp in metric
    if unit_system == "metric" and unit in ("cup","tbsp","tsp"):
        # fall back to metric counterpart
        unit = {"cup":"ml","tbsp":"ml","tsp":"ml"}[unit]
        unit_txt = unit
        qty = max(5, int(qty * (240 if unit=="ml" else 15)))  # rough conversion
        qty_txt = str(qty)
    return f"{qty_txt} {unit_txt} {ing['item']}"

def record_text_view(rec: Dict) -> str:
    # Plain text view (tagged header + blocks)
    tags = rec["tags"]
    head = f"<unit={tags['unit']}> <cuisine={tags['cuisine']}> <diet={tags['diet']}> <lvl={tags['lvl']}>"
    lines = [head, rec["title"], "ingredients:"]
    for ing in rec["ingredients"]:
        lines.append("- " + ingredient_line(ing, tags["unit"]))
    lines.append("steps:")
    for idx, s in enumerate(rec["steps"], 1):
        lines.append(f"{idx}. {s}")
    return "\n".join(lines)

def validate_record(rec: Dict) -> Tuple[bool, List[str]]:
    issues = []
    tags = rec["tags"]
    items = [i["item"] for i in rec["ingredients"]]
    # Diet rules
    if not diet_ok(tags["diet"], items):
        issues.append("diet_violation")
    # Method forbids check
    if method_forbids(tags["method"], items):
        issues.append("method_violation")
    # Unit system adherence
    if tags["unit"] == "metric":
        txt = record_text_view(rec)
        if re.search(r"\b(cup|tbsp|tsp)\b", txt):
            issues.append("unit_system_violation")
    # Step length
    for s in rec["steps"]:
        if len(tokens(s)) > 14:
            issues.append("step_length")
            break
    # Ingredients bounds
    if not (3 <= len(rec["ingredients"]) <= 6):
        issues.append("ingredient_count")
    # Steps bounds
    lb, hb = METHOD_TO_STEP_BOUNDS.get(tags["method"], (3,5))
    if not (lb <= len(rec["steps"]) <= hb):
        issues.append("step_count")
    # Cuisine avoid pairs
    if violates_cuisine_pairs(tags["cuisine"], items):
        issues.append("cuisine_pair")
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
    # Steps and ingredients counts scaled by level
    desired_ing = ing_count_sample()
    if tags["lvl"] >= 4 and desired_ing < 4:
        desired_ing = 4
    # ensure method requirements fit
    ing_list = pick_ingredients(tags["method"], tags["cuisine"], tags["diet"], tags["unit"], desired_ing)
    if not ing_list:
        return None
    # steps
    steps = build_steps(tags["method"], ing_list, tags["cuisine"])
    # title
    items = [i["item"] for i in ing_list]
    title = title_from(tags["method"], items, tags["cuisine"])
    # time & servings
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
        return None
    return rec

# -----------------------------
# Generation loop
# -----------------------------

def generate_dataset(total: int, out_dir: str, shards: int = 100, gzip_out: bool = False,
                     seed: int = 42, noise_rate: float = 0.05, log_every: int = 5000):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    # Dedup via canonical key set (per-run; memory is fine up to ~1–2M keys if needed)
    seen = set()
    per_shard = max(1, total // shards)
    # Adjust shards so that we don't create more shards than needed
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

    shard_idx = 0
    path, fh = writer_handle(shard_idx)

    try:
        while written < total:
            attempts += 1
            rec = assemble_record()
            if not rec:
                continue

            # Optional noise injection (very light): flip 1 rule at tiny rate
            if noise_rate > 0 and random.random() < noise_rate:
                # inject unit mismatch for metric occasionally
                if rec["tags"]["unit"] == "metric":
                    for ing in rec["ingredients"]:
                        if ing["unit"] in ("g","ml") and random.random() < 0.5:
                            ing["unit"] = random.choice(["cup","tbsp","tsp"])
                            break
                # or add odd pair
                elif rec["tags"]["cuisine"] == "italian" and "soy sauce" not in [i["item"] for i in rec["ingredients"]]:
                    rec["ingredients"].append({
                        "item":"soy sauce","category":"liquid","qty":1,"unit":"tbsp"
                    })
                rec["id"] = uuid.uuid4().hex  # new id

            # Validate again (for noise we still allow; but mark violations if any)
            ok, issues = validate_record(rec)
            if not ok:
                # keep a small portion as noisy if issues limited
                if noise_rate > 0 and len(issues) <= 2 and random.random() < 0.5:
                    rec.setdefault("violations", issues)
                else:
                    continue

            key = canonical_key(rec)
            if key in seen:
                continue
            seen.add(key)

            # write jsonl line
            line = json.dumps(rec, ensure_ascii=False)
            fh.write(line + "\n")
            written += 1

            # rotate shard
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

    # Write a small manifest
    manifest = {
        "total": written,
        "shards": shard_idx + 1,
        "per_shard": per_shard,
        "seed": seed,
        "gzip": gzip_out,
        "noise_rate": noise_rate,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"Done. Wrote {written} records across {shard_idx+1} shard(s) to {out_dir}")

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic recipelets dataset")
    ap.add_argument("--total", type=int, default=500000, help="Total number of records to generate")
    ap.add_argument("--out_dir", type=str, default="./recipelets", help="Output directory")
    ap.add_argument("--shards", type=int, default=10, help="Number of shards (files)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--gzip", action="store_true", help="Write .jsonl.gz shards")
    ap.add_argument("--noise_rate", type=float, default=0.05, help="Rate of light noise injection")
    args = ap.parse_args()

    generate_dataset(total=args.total, out_dir=args.out_dir, shards=args.shards,
                     gzip_out=args.gzip, seed=args.seed, noise_rate=args.noise_rate)

if __name__ == "__main__":
    main()
