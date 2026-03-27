"""
Numinous SN6 — v8 (Segment-Routed Ensemble)
=============================================
Major upgrades over v7:
  - Segment routing: geo_strikes / weather / app_store / sports / general
  - Geo-strikes: static anchors + city adjustments (no LLM guessing)
  - App store: incumbent win-rate table + Polymarket slot races
  - Weather: sigmoid recalibration to fix negative bias
  - Polymarket: /public-search events API for exact match (not keyword market search)
  - gpt-4o (upgraded from gpt-4o-mini) with web_search tool
  - Removed crude affirm/negate web signal (noisy, hurts calibration)
  - Weighted geometric pooling with segment-aware weights
"""

import asyncio
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import httpx

# ── env ───────────────────────────────────────────────────────────────────────
GATEWAY   = os.environ.get("SANDBOX_PROXY_URL", "http://sandbox_proxy").rstrip("/")
SESSION   = os.environ.get("RUN_ID", "v8-session")
UTC_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ── endpoints ─────────────────────────────────────────────────────────────────
PATH_CHUTES   = GATEWAY + "/api/gateway/chutes/chat/completions"
PATH_OAI      = GATEWAY + "/api/gateway/openai/responses"
PATH_DESEARCH = GATEWAY + "/api/gateway/desearch/search"
GAMMA_PROXY   = GATEWAY + "/api/gateway/desearch"   # polymarket via crawl fallback
GAMMA_API     = "https://gamma-api.polymarket.com"

# ── models ────────────────────────────────────────────────────────────────────
MODEL_OAI_PRIMARY     = "gpt-4o"
MODEL_CHUTES_PRIMARY  = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_CHUTES_FALLBACK = "deepseek-ai/DeepSeek-V3-0324"

# ── calibration ───────────────────────────────────────────────────────────────
BASE_RATE        = 0.35
PROB_FLOOR       = 0.02
PROB_CEIL        = 0.98
SHRINK_FRACTION  = 0.15

# ── stopwords ─────────────────────────────────────────────────────────────────
_STOP = frozenset(
    "will the a an in on of by to and or for at is it from that "
    "this with as are was be before after ever than least more less "
    "have has had could would should may might must end between".split()
)

# ── segment detection ─────────────────────────────────────────────────────────
_GEO_GENERAL = frozenset({"geopolitics", "politics", "iran", "middle east",
                           "elections", "primaries", "global elections"})
_IRAN_KIN_RE = re.compile(
    r"retaliate with.*strike|launch.*rocket|launch.*attack|fire rocket"
    r"|sirens sound|missile strike.*on|strike.*in response"
    r"|drone strike|missile strike|military strike|artillery strike"
    r"|drone attack|air\s?strike|experience.*strike|experience.*attack", re.I)


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
    if any(m in blob for m in (" vs ", " vs. ", "cricket", "both teams to score")) or " win?" in tl:
        return "sports"
    if "app store" in blob:
        return "app_store"
    if " temperature " in blob:
        return "weather"
    if "earnings" in blob:
        return "earnings"
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


def weather_adjust(p: float) -> float:
    """Sigmoid recalibration that fixes systematic underforecasting on weather events."""
    slope, intercept = 0.15, -1.386
    x = max(-500.0, min(500.0, slope * p + intercept))
    return clamp(1.0 / (1.0 + math.exp(-x)))


# ═══════════════════════════════════════════════════════════════════════════════
# Probability parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_prob(txt: str) -> float | None:
    if not txt:
        return None
    # JSON keys
    for key in ("prediction", "probability", "confidence", "forecast", "prob", "p"):
        hit = re.search(rf'["\']?{key}["\']?\s*:\s*([\d.]+)', txt, re.I)
        if hit:
            v = float(hit.group(1))
            return clamp(v / 100.0 if v > 1.5 else v)
    # Tagged output
    hit = re.search(r'PREDICTION\s*:\s*([\d.]+)', txt, re.I)
    if hit:
        v = float(hit.group(1))
        return clamp(v / 100.0 if v > 1.5 else v)
    # Bare decimal
    decimals = re.findall(r'\b(0\.\d{2,4})\b', txt)
    if decimals:
        return clamp(float(decimals[-1]))
    # Percentage
    pcts = re.findall(r'\b(\d{1,3}(?:\.\d)?)\s*%', txt)
    if pcts:
        return clamp(float(pcts[-1]) / 100.0)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP
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
                print(f"[v8] post_retry failed {url}: {exc}")
            await asyncio.sleep(2 ** i)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Polymarket — events API (exact match approach)
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
    # Try outcomePrices field first
    op = market.get("outcomePrices")
    if op:
        try:
            arr = json.loads(op) if isinstance(op, str) else op
            if isinstance(arr, list) and arr:
                return clamp(float(arr[0]))
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
    # Try tokens
    for tok in market.get("tokens") or []:
        if str(tok.get("outcome", "")).upper() == "YES":
            raw = tok.get("price")
            if raw is not None:
                return clamp(float(raw))
    return None


async def polymarket_signal(client, title: str) -> tuple[float | None, float]:
    """
    Returns (yes_price, match_quality).
    Tries /public-search events API first (exact match), falls back to /markets.
    """
    kw = _keywords(title)

    # Strategy 1: events search (same as top miner)
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
            print(f"[v8] Polymarket (events): {best_price:.3f} quality={best_quality:.2f}")
            return best_price, best_quality
    except Exception as e:
        print(f"[v8] Polymarket events API error: {e}")

    # Strategy 2: /markets fallback
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
                    print(f"[v8] Polymarket (markets): {p:.3f} quality={q:.2f}")
                    return p, q
    except Exception as e:
        print(f"[v8] Polymarket markets API error: {e}")

    return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Geo-strikes specialist
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_RULES = [
    (re.compile(r"more than \d+(?:\s+\S+){0,3}\s+drone", re.I),    "more_than_N_drones",  0.96),
    (re.compile(r"more than \d+(?:\s+\S+){0,3}\s+strike", re.I),   "more_than_N_strikes", 0.99),
    (re.compile(r"without any strike|strike-free|no strikes", re.I),"strike_free",         0.01),
    (re.compile(r"energy grid.*directly hit", re.I),                 "energy_grid_hit",     0.08),
    (re.compile(r"port.*(?:infrastructure|harbor)|shipyard.*target", re.I), "port_infra",  0.01),
    (re.compile(r"(?:railway|bridge).*targeted|hospital.*(?:hit|strike)", re.I), "other_infra", 0.01),
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

# Incumbent table: (store_type, rank) -> (primary_app, win_rate)
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

    # Check secondary incumbents
    for sec_app, sec_rate in _APP_SECONDARY.get(key, []):
        if app == sec_app.lower():
            return sec_rate

    # Unknown app at this slot
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


async def oai_signal(client, question: str, cutoff: str) -> float | None:
    prompt = (
        f"Forecasting question: {question}\n"
        f"Resolution cutoff: {cutoff}\n"
        f"Today: {UTC_TODAY}\n\n"
        "Search for current information, apply reference-class forecasting, then output:\n"
        "PREDICTION: [0.01-0.99]\n"
        "REASONING: [your analysis]"
    )
    data = await post_retry(
        client, PATH_OAI,
        {
            "model": MODEL_OAI_PRIMARY,
            "input": [{"role": "user", "content": prompt}],
            "instructions": _SYS_GENERAL,
            "tools": [{"type": "web_search_preview"}],
            "max_output_tokens": 400,
            "metadata": {"session": SESSION},
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
        f"Cutoff: {cutoff}\nToday: {UTC_TODAY}\n\n"
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
        },
        timeout=90, attempts=3,
    )
    if not data:
        return None
    txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return parse_prob(txt)


# ═══════════════════════════════════════════════════════════════════════════════
# Routing logic
# ═══════════════════════════════════════════════════════════════════════════════

async def run_geo_strikes(record: dict) -> float:
    title = record.get("title", "")
    anchor, qtype = geo_anchor(title)
    print(f"[v8] geo_strikes anchor={anchor:.2f} qtype={qtype}")
    # Static anchor is primary signal — no LLM for geo
    return anchor


async def run_iran_diplomatic(_record: dict) -> float:
    print("[v8] iran_diplomatic → 0.05")
    return 0.05


async def run_app_store(record: dict, client) -> float:
    title = record.get("title", "")
    prior = app_incumbent_prior(title)
    if prior is not None:
        print(f"[v8] app_store incumbent prior={prior:.2f}")
        # Try to get Polymarket slot data to refine
        pm_price, pm_q = await polymarket_signal(client, title)
        if pm_price is not None and pm_q >= 0.30:
            result = geo_pool([prior, pm_price], [1.5, pm_q * 3.0])
            print(f"[v8] app_store pool prior={prior:.2f} pm={pm_price:.2f} → {result:.3f}")
            return result
        return prior
    # Unknown app — fall through to general
    print("[v8] app_store unknown app → general flow")
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
    print(f"[v8] weather raw={raw:.3f} sigmoid_adjusted={adjusted:.3f}")
    return adjusted


async def run_general(record: dict, client) -> float:
    title  = record.get("title", "")
    desc   = record.get("description", "") or ""
    cutoff = str(record.get("cutoff", ""))

    # Parallel: Polymarket + OAI web search
    pm_task  = asyncio.create_task(polymarket_signal(client, title))
    oai_task = asyncio.create_task(oai_signal(client, title, cutoff))

    pm_price, pm_q = await pm_task
    oai_est        = await oai_task

    # Fast path: near-certain Polymarket match
    if pm_price is not None and pm_q >= 0.65 and (pm_price <= 0.05 or pm_price >= 0.95):
        print(f"[v8] fast-path Polymarket {pm_price:.3f}")
        return pm_price

    # Build context for Chutes
    ctx_parts = []
    if pm_price is not None:
        ctx_parts.append(f"Polymarket YES price: {pm_price:.3f} (match quality: {pm_q:.2f})")
    if oai_est is not None:
        ctx_parts.append(f"OpenAI web search estimate: {oai_est:.3f}")
    if desc:
        ctx_parts.append(f"Description: {desc[:500]}")
    context = "\n\n".join(ctx_parts) or "No structured data available."

    chutes_est = await chutes_signal(client, title, context, cutoff)
    if chutes_est is None:
        chutes_est = await chutes_signal(client, title, context, cutoff, model=MODEL_CHUTES_FALLBACK)

    print(f"[v8] signals — PM: {pm_price}, OAI: {oai_est}, Chutes: {chutes_est}")

    # Weighted pool
    votes, weights = [], []
    if pm_price is not None:
        votes.append(pm_price);    weights.append(4.0 * pm_q)
    if oai_est is not None:
        votes.append(oai_est);     weights.append(2.0)
    if chutes_est is not None:
        votes.append(chutes_est);  weights.append(1.5)

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
    """Required entry point for Numinous SN6 validator sandbox."""

    eid     = event_data.get("event_id", "unknown")
    title   = event_data.get("title", "")
    segment = classify_segment(event_data)

    print(f"[v8] segment={segment} title={title[:80]}")

    # Normalise cutoff timestamp
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
        # Geo and Iran diplomatic don't need an HTTP client
        if segment == "geo_strikes":
            return await run_geo_strikes(event_data)
        if segment == "iran_diplomatic":
            return await run_iran_diplomatic(event_data)

        async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
            if segment == "app_store":
                return await run_app_store(event_data, client)
            if segment == "weather":
                return await run_weather(event_data, client)
            return await run_general(event_data, client)

    try:
        prob = asyncio.run(_run())
    except Exception as exc:
        print(f"[v8] Fatal error: {exc}")
        prob = BASE_RATE

    prob = clamp(prob)
    print(f"[v8] Final prediction: {prob:.4f}")

    return {
        "event_id":   eid,
        "prediction": prob,
        "reasoning":  f"v8 segment={segment} prediction={prob:.3f}",
    }
