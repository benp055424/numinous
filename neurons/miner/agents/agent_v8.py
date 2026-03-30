"""
Numinous SN6 — v8.1 (Optimized Ensemble)
==========================================
Major changes from v8.0 based on competitive intelligence (top 5 miners analysis):

  Bias fixes (measured -0.048 negative bias):
  - BASE_RATE 0.35 → 0.43 (all top miners have positive bias)
  - SHRINK_FRACTION 0.15 → 0.10
  - Post-ensemble debiasing +0.04 in logit space
  - iran_diplomatic 0.05 → 0.15

  Model upgrades (matching #1 miner SirMutton):
  - OpenAI: gpt-4o → gpt-5-mini (used by #1, #3, #5 miners)
  - Added OpenRouter Claude fallback (used by #5 miner)
  - Chutes primary unchanged (Qwen3-235B proven good)

  Technique upgrades (from #1 SirMutton + #5 flynn0509):
  - Extremizing transform (Satopää et al.) after ensemble pooling
  - Indicia OSINT (LiveUAMap) for geo events (used by #1, free)
  - Polymarket via Desearch crawl (more reliable in sandbox, used by #1 and #5)
  - Sphere-based search hint generation (from #1)
  - Stronger probability parser (handles more output formats)

  Weather fix:
  - Broken sigmoid (mapped 0.5→0.21) replaced with logit-space shift

  Pool strategy (current weights: Global 55%, Geo 20%, Finance 15%, Sports 5%, Crypto 5%):
  - Geo pool (20%) is key target — Indicia + anchors + LLM leash
  - Finance pool (15%) — Polymarket passthrough for scored-against-PM events
  - Global pool (55%) — best overall Brier across all events
"""

import asyncio
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

import httpx

# ── env ───────────────────────────────────────────────────────────────────────
GATEWAY = os.environ.get("SANDBOX_PROXY_URL", "http://sandbox_proxy").rstrip("/")
RUN_ID  = os.environ.get("RUN_ID") or str(uuid4())

# ── endpoints ─────────────────────────────────────────────────────────────────
PATH_OAI       = GATEWAY + "/api/gateway/openai/responses"
PATH_CHUTES    = GATEWAY + "/api/gateway/chutes/chat/completions"
PATH_OPENROUTER = GATEWAY + "/api/gateway/openrouter/chat/completions"
PATH_DESEARCH  = GATEWAY + "/api/gateway/desearch/search"
PATH_CRAWL     = GATEWAY + "/api/gateway/desearch/web/crawl"
PATH_INDICIA   = GATEWAY + "/api/gateway/numinous-indicia"
GAMMA_API      = "https://gamma-api.polymarket.com"

# ── models ────────────────────────────────────────────────────────────────────
MODEL_OAI_PRIMARY      = "gpt-5-mini"
MODEL_CHUTES_PRIMARY   = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_CHUTES_FALLBACK  = "deepseek-ai/DeepSeek-V3-0324"
MODEL_OPENROUTER       = "anthropic/claude-sonnet-4-6"

# ── calibration ───────────────────────────────────────────────────────────────
BASE_RATE       = 0.43
PROB_FLOOR      = 0.02
PROB_CEIL       = 0.98
SHRINK_FRACTION = 0.10
DEBIAS_LOGIT    = 0.16
EXTREMIZE_D     = 1.2

# ── stopwords ─────────────────────────────────────────────────────────────────
_STOP = frozenset(
    "will the a an in on of by to and or for at is it from that "
    "this with as are was be before after ever than least more less "
    "have has had could would should may might must end between".split()
)

# ── timing ────────────────────────────────────────────────────────────────────
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time else 0

def _budget_left():
    return max(0, 220 - _elapsed())


# ═══════════════════════════════════════════════════════════════════════════════
# Segment detection
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_GENERAL = frozenset({"geopolitics", "politics", "iran", "middle east",
                           "elections", "primaries", "global elections",
                           "world elections", "regional spillover"})
_IRAN_KIN_RE = re.compile(
    r"retaliate with.*strike|launch.*rocket|launch.*attack|fire rocket"
    r"|sirens sound|missile strike.*on|strike.*in response"
    r"|drone strike|missile strike|military strike|artillery strike"
    r"|drone attack|air\s?strike|experience.*strike|experience.*attack", re.I)

_SPHERES = (
    ("athletics", frozenset({"MATCH", "GAME", "WIN", "VS", "CHAMPIONSHIP", "LEAGUE", "CUP", "PLAYOFF", "GOAL", "SCORE"})),
    ("governance", frozenset({"ELECTION", "VOTE", "POLL", "PRESIDENT", "GOVERNOR", "SENATOR", "MAYOR", "TARIFF", "SANCTION", "WAR", "TREATY"})),
    ("financial", frozenset({"RATE", "FED", "GDP", "INFLATION", "BITCOIN", "CRYPTO", "PRICE", "MARKET", "$", "STOCK"})),
    ("technology", frozenset({"LAUNCH", "RELEASE", "APP", "SOFTWARE", "UPDATE", "SHIP", "ANNOUNCE"})),
    ("showbusiness", frozenset({"MOVIE", "FILM", "OSCAR", "GRAMMY", "EMMY", "ALBUM", "BOX OFFICE", "AWARD"})),
)

_SPHERE_HINTS = {
    "athletics": ("betting odds", "injuries", "recent form", "head-to-head", "standings"),
    "governance": ("Polymarket", "polling", "official statement", "Reuters AP"),
    "financial": ("FedWatch", "central bank", "market expectations", "economic data"),
    "technology": ("official blog", "press release", "SEC filing", "launch date"),
    "showbusiness": ("box office", "reviews", "awards predictions", "release date"),
    "general": ("Polymarket", "recent news", "official source"),
}


def _get_sphere(record: dict) -> str:
    blob = (record.get("title", "") + " " + record.get("description", "")).upper()
    for label, terms in _SPHERES:
        for term in terms:
            if term in blob:
                return label
    return "general"


def classify_segment(record: dict) -> str:
    meta   = record.get("metadata", {})
    topics = {t.lower() for t in meta.get("topics", [])}
    title  = record.get("title", "")
    blob   = (title + " " + record.get("description", "")).lower()
    tl     = title.lower()

    if "strikes" in topics or (
        not (topics & _GEO_GENERAL)
        and re.search(r"military strike|missile strike|drone (?:attack|strike)|air\s?strike", blob)
    ):
        return "geo_strikes"

    if topics == {"geopolitics", "iran"} and not _IRAN_KIN_RE.search(title):
        return "iran_diplomatic"

    for seg in ("sports", "app store", "weather", "earnings", "election", "inflation", "price"):
        if seg in topics:
            return seg

    if "election" in blob:
        return "election"
    if any(m in blob for m in (" vs ", " vs. ", "cricket", "both teams to score", "upcoming game", "stoppage time")) or " win?" in tl or " win " in tl:
        return "sports"
    if "app store" in blob or " app " in blob:
        return "app_store"
    if " temperature " in blob:
        return "weather"
    if "earnings" in blob or (any(q in blob for q in ("q1", "q2", "q3", "q4")) and "above" in blob):
        return "earnings"
    if "inflation" in blob:
        return "inflation"
    if " price of " in blob:
        return "price"
    return "general"


# ═══════════════════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def clamp(v) -> float:
    try:
        return max(PROB_FLOOR, min(PROB_CEIL, float(v)))
    except (TypeError, ValueError):
        return BASE_RATE


def logit(p: float) -> float:
    p = clamp(p)
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def geo_pool(votes: list, weights: list) -> float:
    total_w = log_sum = 0.0
    for p, w in zip(votes, weights):
        if p is not None and w > 0:
            log_sum += w * logit(p)
            total_w += w
    if total_w == 0:
        return BASE_RATE
    return clamp(inv_logit(log_sum / total_w))


def shrink(p: float, frac: float = SHRINK_FRACTION) -> float:
    return clamp((1.0 - frac) * p + frac * BASE_RATE)


def extremize(p: float, d: float = EXTREMIZE_D) -> float:
    if d == 1.0:
        return p
    return clamp(inv_logit(d * logit(p)))


def debias(p: float) -> float:
    return clamp(inv_logit(logit(p) + DEBIAS_LOGIT))


def weather_adjust(p: float) -> float:
    return clamp(inv_logit(logit(p) + 0.30))


# ═══════════════════════════════════════════════════════════════════════════════
# Probability parsing (enhanced — handles more formats from various LLMs)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_prob(txt: str) -> float | None:
    if not txt:
        return None
    for key in ("prediction", "probability", "confidence", "forecast", "prob", "p", "likelihood", "final_probability"):
        hit = re.search(rf'["\']?{key}["\']?\s*:\s*([\d.]+)', txt, re.I)
        if hit:
            v = float(hit.group(1))
            return clamp(v / 100.0 if v > 1.5 else v)
    hit = re.search(r'PREDICTION\s*:\s*([\d.]+)', txt, re.I)
    if hit:
        v = float(hit.group(1))
        return clamp(v / 100.0 if v > 1.5 else v)
    hit = re.search(r'(?:estimate|forecast|likelihood)\s*:\s*(?:is\s+)?([\d.]+)', txt, re.I)
    if hit:
        v = float(hit.group(1))
        return clamp(v / 100.0 if v > 1.5 else v)
    decimals = re.findall(r'\b(0\.\d{2,4})\b', txt)
    if decimals:
        return clamp(float(decimals[-1]))
    pcts = re.findall(r'\b(\d{1,3}(?:\.\d)?)\s*%', txt)
    if pcts:
        return clamp(float(pcts[-1]) / 100.0)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════════════════

async def post_retry(client, url, body, timeout=90, attempts=3):
    for i in range(attempts):
        try:
            r = await client.post(url, json=body, timeout=timeout)
            if r.status_code in (429, 500, 502, 503):
                await asyncio.sleep(min(2 ** i * 2, 20))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if i == attempts - 1:
                print(f"[v8.1] post_retry failed {url}: {exc}")
            await asyncio.sleep(2 ** i)
    return None


def crawl_sync(url, timeout=25.0):
    for attempt in range(3):
        try:
            resp = httpx.post(PATH_CRAWL, json={"url": url, "run_id": RUN_ID}, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", "")
            if not content:
                return None
            return json.loads(content) if isinstance(content, str) else content
        except Exception:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Polymarket — via Desearch crawl (more reliable in sandbox, used by #1 miner)
# ═══════════════════════════════════════════════════════════════════════════════

def _jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z0-9]+", a.lower())) - _STOP
    wb = set(re.findall(r"[a-z0-9]+", b.lower())) - _STOP
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _keywords(title: str, n: int = 8) -> str:
    return " ".join(
        w for w in re.findall(r"[a-zA-Z0-9]+", title)[:n]
        if w.lower() not in _STOP
    )


def _yes_price(market: dict) -> float | None:
    op = market.get("outcomePrices")
    if op:
        try:
            arr = json.loads(op) if isinstance(op, str) else op
            if isinstance(arr, list) and arr:
                return clamp(float(arr[0]))
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
    for tok in market.get("tokens") or []:
        if str(tok.get("outcome", "")).upper() == "YES":
            raw = tok.get("price")
            if raw is not None:
                return clamp(float(raw))
    return None


async def polymarket_signal(client, title: str) -> tuple[float | None, float]:
    kw = _keywords(title)

    # Strategy 1: events search via direct API (fast)
    try:
        r = await client.get(
            GAMMA_API + "/public-search",
            params={"q": kw},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        events = data.get("events") or []
        best_price, best_quality = None, 0.0
        for ev in events:
            for m in ev.get("markets") or []:
                q = _jaccard(title, m.get("question", ""))
                if q > best_quality:
                    p = _yes_price(m)
                    if p is not None:
                        best_quality = q
                        best_price = p
        if best_price is not None and best_quality >= 0.25:
            print(f"[v8.1] PM (events): {best_price:.3f} q={best_quality:.2f}")
            return best_price, best_quality
    except Exception as e:
        print(f"[v8.1] PM events error: {e}")

    # Strategy 2: /markets fallback via direct API
    try:
        r = await client.get(
            GAMMA_API + "/markets",
            params={"limit": 10, "active": "true", "closed": "false", "search": kw},
            timeout=14,
        )
        r.raise_for_status()
        markets = r.json()
        if isinstance(markets, list) and markets:
            ranked = sorted(markets, key=lambda m: _jaccard(title, m.get("question", "")), reverse=True)
            top = ranked[0]
            q = _jaccard(title, top.get("question", ""))
            if q >= 0.25:
                p = _yes_price(top)
                if p is not None:
                    print(f"[v8.1] PM (markets): {p:.3f} q={q:.2f}")
                    return p, q
    except Exception as e:
        print(f"[v8.1] PM markets error: {e}")

    # Strategy 3: Desearch crawl fallback (used by #1 miner as primary)
    try:
        url = f"{GAMMA_API}/public-search?q={kw}"
        data = crawl_sync(url, timeout=20.0)
        if data:
            events = data.get("events") or []
            best_price, best_quality = None, 0.0
            for ev in events:
                for m in ev.get("markets") or []:
                    q = _jaccard(title, m.get("question", ""))
                    if q > best_quality:
                        p = _yes_price(m)
                        if p is not None:
                            best_quality = q
                            best_price = p
            if best_price is not None and best_quality >= 0.25:
                print(f"[v8.1] PM (crawl): {best_price:.3f} q={best_quality:.2f}")
                return best_price, best_quality
    except Exception as e:
        print(f"[v8.1] PM crawl error: {e}")

    return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Indicia OSINT (from #1 miner — LiveUAMap + X-OSINT for geo events)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_indicia(client, city: str | None, title: str) -> str:
    signals = []
    for endpoint in ("/liveuamap", "/x-osint"):
        try:
            r = await client.post(
                PATH_INDICIA + endpoint,
                json={"run_id": RUN_ID, "limit": 20},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            for s in data.get("signals", []):
                signals.append(
                    f"[{s.get('category','')}] {s.get('signal','')} "
                    f"(confidence={s.get('confidence','')}, {s.get('timestamp','')})"
                )
        except Exception as e:
            print(f"[v8.1] Indicia {endpoint} error: {e}")

    if not signals:
        return ""

    text = "\n".join(signals[:30])
    if city:
        city_lower = city.lower()
        relevant = [s for s in signals if city_lower in s.lower()]
        if relevant:
            text = f"Signals mentioning {city}:\n" + "\n".join(relevant[:15]) + "\n\nAll signals:\n" + "\n".join(signals[:15])
        else:
            text = f"No signals mention {city} specifically.\n\nAll signals:\n" + "\n".join(signals[:20])

    print(f"[v8.1] Indicia: {len(signals)} signals")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Geo-strikes specialist
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_RULES = [
    (re.compile(r"more than \d+(?:\s+\S+){0,3}\s+drone", re.I),    "more_than_N_drones",  0.96),
    (re.compile(r"more than \d+(?:\s+\S+){0,3}\s+strike", re.I),   "more_than_N_strikes", 0.99),
    (re.compile(r"without any strike|strike-free|strike free|hours without|no strikes", re.I), "strike_free", 0.01),
    (re.compile(r"energy grid.*directly hit", re.I),                 "energy_grid_hit",     0.08),
    (re.compile(r"port.*(?:infrastructure|harbor|shipyard)|(?:harbor|shipyard).*targeted", re.I), "port_infra", 0.01),
    (re.compile(r"(?:railway|bridge).*(?:infrastructure|targeted|directly hit)|(?:strike|hit).*hospital|hospital.*(?:hit|strike|targeted)", re.I), "other_infra", 0.01),
    (re.compile(r"(?:energy|power).*(?:infrastructure|targeted)", re.I), "energy_infra",  0.09),
    (re.compile(r"residential", re.I),                               "residential",        0.05),
    (re.compile(r"drone strike", re.I),                              "drone_strike",        0.89),
    (re.compile(r"missile strike", re.I),                            "missile_strike",      0.81),
    (re.compile(r"military strike", re.I),                           "military_strike",     0.68),
    (re.compile(r"artillery strike", re.I),                          "artillery_strike",    0.58),
    (re.compile(r"drone attack", re.I),                              "drone_attack",        0.55),
]

_GEO_CITY_ADJ = {
    "kharkiv": +0.10, "odesa": +0.10, "dnipro": +0.08,
    "kramatorsk": +0.05, "donetsk": +0.05, "kherson": +0.05,
    "sumy": +0.03, "poltava": +0.03, "kyiv": +0.03,
    "zaporizhzhia": +0.02,
    "kursk": -0.20, "volgograd": -0.18, "belgorod": -0.10,
    "lviv": -0.10, "bryansk": -0.08,
}
_GEO_CITY_OK = frozenset({"military_strike", "drone_strike", "missile_strike",
                           "drone_attack", "artillery_strike"})
_GEO_LEASH = 0.25


def _geo_city(title: str) -> str | None:
    m = re.match(r"Will\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+experience", title)
    if m:
        return m.group(1).split(",")[0].strip()
    m = re.search(r"(?:in|on|at|near)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", title)
    if m:
        c = m.group(1).split(",")[0].strip()
        if c.lower() not in ("unknown", "the", "ukraine"):
            return c
    return None


def geo_anchor(title: str) -> tuple[float, str]:
    for regex, qtype, base in _GEO_RULES:
        if regex.search(title):
            anchor = base
            city = _geo_city(title)
            if city and qtype in _GEO_CITY_OK:
                adj = _GEO_CITY_ADJ.get(city.lower(), 0.0)
                anchor = max(0.01, min(0.99, anchor + adj))
            return anchor, qtype
    return 0.50, "unmatched"


def geo_leash(prob: float, anchor: float) -> float:
    lo = max(0.01, anchor - _GEO_LEASH)
    hi = min(0.99, anchor + _GEO_LEASH)
    return clamp(max(lo, min(hi, prob)))


# ═══════════════════════════════════════════════════════════════════════════════
# App store specialist
# ═══════════════════════════════════════════════════════════════════════════════

_APP_INCUMBENTS = {
    ("Free",  1): ("ChatGPT",               0.92),
    ("Free",  2): ("Google Gemini",         0.24),
    ("Free",  3): ("Google Gemini",         0.24),
    ("Free",  4): ("Threads",               0.36),
    ("Paid",  1): ("Shadowrocket",          0.92),
    ("Paid",  2): ("HotSchedules",          0.97),
    ("Paid",  3): ("AnkiMobile Flashcards", 0.78),
    ("Paid",  4): ("Procreate Pocket",      0.83),
}

_APP_SECONDARY = {
    ("Free",  2): [("Freecash", 0.28)],
    ("Free",  3): [("Threads",  0.29)],
    ("Free",  4): [("Google Gemini", 0.25)],
    ("Paid",  3): [("Procreate Pocket", 0.13)],
    ("Paid",  4): [("AnkiMobile", 0.17)],
}

_MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}


def parse_app_title(title: str) -> dict | None:
    m = re.search(
        r"Will (.+?) be (?:the )?#(\d+) (Free|Paid) [Aa]pp in the US "
        r"(?:iPhone |Apple )?App Store on (\w+ \d+)",
        title, re.I,
    )
    if not m:
        return None
    app, rank, typ, ds = m.groups()
    bits = ds.split()
    mo = _MONTHS.get(bits[0].lower())
    if not mo or len(bits) < 2:
        return None
    try:
        day = int(bits[1])
        date = datetime(2026, mo, day).date()
    except ValueError:
        return None
    return {"app": app.strip(), "rank": int(rank), "type": typ.capitalize(), "date": date}


def app_incumbent_prior(title: str) -> float | None:
    parsed = parse_app_title(title)
    if not parsed:
        return None
    app  = parsed["app"].lower()
    rank = parsed["rank"]
    typ  = parsed["type"]

    key = (typ, rank)
    primary, win_rate = _APP_INCUMBENTS.get(key, (None, None))

    if primary and app == primary.lower():
        return win_rate

    for sec_app, sec_rate in _APP_SECONDARY.get(key, []):
        if app == sec_app.lower():
            return sec_rate

    return 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# LLM signals
# ═══════════════════════════════════════════════════════════════════════════════

_SYS_GENERAL = """\
You are an expert forecaster for prediction markets. Estimate P(YES) with rigorous research.
CRITICAL: You MUST complete ALL your research in a SINGLE web_search tool call. Pack all your queries into one search. After receiving results, respond immediately.

RESEARCH APPROACH — adapt your search strategy to the event type:
For competitions/matches: Search betting odds (decimal odds D → prob ≈ 1/D minus margin), recent form, injuries.
For political events: Search Polymarket/PredictIt first — market prices are strong signals. Check polling aggregates.
For economic events: Search market-implied probabilities (CME FedWatch for rates, futures markets).
For product launches: Check official company channels, press releases.

ALWAYS DO:
1. Search "Polymarket [topic]" — if market exists, price ≈ probability
2. Search recent news (last 48-72 hours)
3. Consider time until cutoff (more time = more uncertainty)

ANALYSIS:
- Polymarket price is your anchor — deviate only with strong contrary evidence
- Base rates: how often do similar events happen?
- Resolution criteria are literal — read exact wording carefully
- Range: never return exactly 0 or 1, use [0.01, 0.99]

OUTPUT FORMAT:
PREDICTION: [0.01-0.99]
REASONING: [Key evidence, market signal if found, 3-5 sentences]"""

_SYS_CHUTES = """\
You are a calibrated superforecaster. Given a forecasting question and evidence, produce a single calibrated probability.

Forecasting discipline:
1. Start with the outside view: what is the base rate for this class of event?
2. Adjust for inside view: specific evidence distinguishing this case.
3. Avoid overconfidence: most binary events do not occur.
4. Apply calibration: 70% confidence should be right ~70% of the time.

Output format — respond with ONLY a valid JSON object:
{"prediction": 0.XX}"""

_SYS_GEO = """\
You are an expert forecaster for military conflict events. Estimate P(YES).
You are given a STATIC ANCHOR based on historical resolution rates for this question type.
The anchor is your PRIMARY signal — it reflects the base rate from hundreds of similar events.
You also receive live conflict data from OSINT sources.
RULES:
- Default to the anchor unless live data provides CONCRETE evidence for today specifically.
- 'No signals for location' may mean lower activity — consider lowering prediction.
- 'Multiple confirmed strikes today' — consider raising prediction.
- Vague news articles are NOT sufficient reason to deviate from anchor.
- Never return exactly 0 or 1, use [0.01, 0.99].
OUTPUT FORMAT:
PREDICTION: [0.01-0.99]
REASONING: [Anchor value, live data assessment, adjustment rationale, 2-3 sentences]"""


async def oai_signal(client, question: str, cutoff: str, system: str = _SYS_GENERAL, tools=None) -> float | None:
    if tools is None:
        tools = [{"type": "web_search_preview"}]
    prompt = (
        f"Forecasting question: {question}\n"
        f"Resolution cutoff: {cutoff}\n\n"
        "Search for current information, apply reference-class forecasting, then output:\n"
        "PREDICTION: [0.01-0.99]\n"
        "REASONING: [your analysis]"
    )
    data = await post_retry(
        client, PATH_OAI,
        {
            "model": MODEL_OAI_PRIMARY,
            "input": [{"role": "user", "content": prompt}],
            "instructions": system,
            "tools": tools,
            "max_output_tokens": 400,
            "run_id": RUN_ID,
            "reasoning": {"effort": "medium"},
        },
        timeout=90, attempts=2,
    )
    if not data:
        return None
    for block in data.get("output") or []:
        if block.get("type") == "message":
            for part in block.get("content") or []:
                p = parse_prob(part.get("text", ""))
                if p is not None:
                    return p
    return None


async def chutes_signal(client, question: str, context: str, cutoff: str,
                         model: str = MODEL_CHUTES_PRIMARY) -> float | None:
    user_msg = (
        f"Question: {question}\n"
        f"Cutoff: {cutoff}\n\n"
        f"Evidence:\n{context[:3000]}\n\n"
        'Respond ONLY with: {"prediction": 0.XX}'
    )
    data = await post_retry(
        client, PATH_CHUTES,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYS_CHUTES},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 150,
            "temperature": 0,
            "stream": False,
            "run_id": RUN_ID,
        },
        timeout=90, attempts=3,
    )
    if not data:
        return None
    txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return parse_prob(txt)


async def openrouter_signal(client, question: str, context: str, cutoff: str) -> float | None:
    if _budget_left() < 30:
        return None
    user_msg = (
        f"Forecasting question: {question}\n"
        f"Resolution cutoff: {cutoff}\n\n"
        f"Evidence:\n{context[:2000]}\n\n"
        "Estimate the probability this event resolves YES. "
        'Respond with ONLY: {"prediction": 0.XX}'
    )
    data = await post_retry(
        client, PATH_OPENROUTER,
        {
            "model": MODEL_OPENROUTER,
            "messages": [
                {"role": "system", "content": _SYS_CHUTES},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
            "run_id": RUN_ID,
        },
        timeout=60, attempts=2,
    )
    if not data:
        return None
    txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return parse_prob(txt)


# ═══════════════════════════════════════════════════════════════════════════════
# Routing logic
# ═══════════════════════════════════════════════════════════════════════════════

async def run_geo_strikes(record: dict, client) -> float:
    title = record.get("title", "")
    cutoff = str(record.get("cutoff", ""))
    anchor, qtype = geo_anchor(title)
    city = _geo_city(title)
    print(f"[v8.1] geo_strikes anchor={anchor:.2f} qtype={qtype} city={city}")

    indicia_data = await fetch_indicia(client, city, title)

    if indicia_data:
        geo_prompt = (
            f"EVENT TO FORECAST:\nTitle: {title}\n"
            f"Cutoff: {cutoff}\n\n"
            f"STATIC ANCHOR (primary signal): {anchor:.2f} (type={qtype})\n\n"
            f"LIVE CONFLICT DATA:\n{indicia_data[:3000]}"
        )
        data = await post_retry(
            client, PATH_OAI,
            {
                "model": MODEL_OAI_PRIMARY,
                "input": [{"role": "user", "content": geo_prompt}],
                "instructions": _SYS_GEO,
                "max_output_tokens": 300,
                "run_id": RUN_ID,
            },
            timeout=60, attempts=2,
        )
        if data:
            for block in data.get("output") or []:
                if block.get("type") == "message":
                    for part in block.get("content") or []:
                        p = parse_prob(part.get("text", ""))
                        if p is not None:
                            leashed = geo_leash(p, anchor)
                            print(f"[v8.1] geo LLM+Indicia → {p:.3f} leashed={leashed:.3f}")
                            return leashed

    return anchor


async def run_iran_diplomatic(_record: dict) -> float:
    print("[v8.1] iran_diplomatic → 0.15")
    return 0.15


async def run_app_store(record: dict, client) -> float:
    title = record.get("title", "")
    prior = app_incumbent_prior(title)
    if prior is not None:
        print(f"[v8.1] app_store incumbent prior={prior:.2f}")
        pm_price, pm_q = await polymarket_signal(client, title)
        if pm_price is not None and pm_q >= 0.30:
            result = geo_pool([prior, pm_price], [1.5, pm_q * 3.0])
            print(f"[v8.1] app_store pool prior={prior:.2f} pm={pm_price:.2f} → {result:.3f}")
            return result
        return prior
    print("[v8.1] app_store unknown app → general flow")
    return await run_general(record, client)


async def run_weather(record: dict, client) -> float:
    title  = record.get("title", "")
    cutoff = str(record.get("cutoff", ""))
    pm_price, pm_q = await polymarket_signal(client, title)
    if pm_price is not None and pm_q >= 0.50:
        raw = pm_price
    else:
        oai = await oai_signal(client, title, cutoff)
        raw = oai if oai is not None else BASE_RATE
    adjusted = weather_adjust(raw)
    print(f"[v8.1] weather raw={raw:.3f} adjusted={adjusted:.3f}")
    return adjusted


async def run_general(record: dict, client) -> float:
    title  = record.get("title", "")
    desc   = record.get("description", "") or ""
    cutoff = str(record.get("cutoff", ""))

    pm_task  = asyncio.create_task(polymarket_signal(client, title))
    oai_task = asyncio.create_task(oai_signal(client, title, cutoff))

    pm_price, pm_q = await pm_task
    oai_est        = await oai_task

    if pm_price is not None and pm_q >= 0.65 and (pm_price <= 0.05 or pm_price >= 0.95):
        print(f"[v8.1] fast-path PM {pm_price:.3f}")
        return pm_price

    ctx_parts = []
    if pm_price is not None:
        ctx_parts.append(f"Polymarket YES price: {pm_price:.3f} (match quality: {pm_q:.2f})")
    if oai_est is not None:
        ctx_parts.append(f"OpenAI web search estimate: {oai_est:.3f}")
    if desc:
        ctx_parts.append(f"Description: {desc[:500]}")
    context = "\n\n".join(ctx_parts) or "No structured data available."

    chutes_task = asyncio.create_task(chutes_signal(client, title, context, cutoff))
    or_task = asyncio.create_task(openrouter_signal(client, title, context, cutoff))

    chutes_est = await chutes_task
    if chutes_est is None:
        chutes_est = await chutes_signal(client, title, context, cutoff, model=MODEL_CHUTES_FALLBACK)

    or_est = await or_task

    print(f"[v8.1] signals — PM:{pm_price} OAI:{oai_est} Chutes:{chutes_est} OR:{or_est}")

    votes, weights = [], []
    if pm_price is not None:
        votes.append(pm_price);    weights.append(4.0 * pm_q)
    if oai_est is not None:
        votes.append(oai_est);     weights.append(2.5)
    if chutes_est is not None:
        votes.append(chutes_est);  weights.append(1.5)
    if or_est is not None:
        votes.append(or_est);      weights.append(1.0)

    if not votes:
        return BASE_RATE

    result = geo_pool(votes, weights)
    if len(votes) < 2:
        result = shrink(result, frac=SHRINK_FRACTION * 2)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def agent_main(event_data: dict) -> dict:
    global _start_time
    _start_time = time.time()

    eid     = event_data.get("event_id", "unknown")
    title   = event_data.get("title", "")
    segment = classify_segment(event_data)

    print(f"[v8.1] segment={segment} title={title[:80]}")

    raw_cutoff = event_data.get("cutoff")
    if isinstance(raw_cutoff, str):
        try:
            event_data["cutoff"] = (
                datetime.fromisoformat(raw_cutoff.replace("Z", "+00:00"))
                .strftime("%Y-%m-%d %H:%M UTC")
            )
        except Exception:
            pass

    async def _run() -> float:
        if segment == "iran_diplomatic":
            return await run_iran_diplomatic(event_data)

        async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
            if segment == "geo_strikes":
                return await run_geo_strikes(event_data, client)
            if segment == "app_store":
                return await run_app_store(event_data, client)
            if segment == "weather":
                return await run_weather(event_data, client)
            return await run_general(event_data, client)

    try:
        prob = asyncio.run(_run())
    except Exception as exc:
        print(f"[v8.1] Fatal error: {exc}")
        prob = BASE_RATE

    prob = clamp(prob)
    prob = extremize(prob, EXTREMIZE_D)
    prob = debias(prob)
    prob = clamp(prob)

    elapsed = time.time() - _start_time
    print(f"[v8.1] Final: {prob:.4f} ({elapsed:.1f}s)")

    return {
        "event_id":   eid,
        "prediction": prob,
        "reasoning":  f"v8.1 seg={segment} p={prob:.3f}",
    }
