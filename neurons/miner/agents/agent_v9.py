"""
Numinous SN6 — v9.0 (Reliability-First)
=========================================
Design principles (from leaderboard analysis of current top miners):

1. NEVER miss a prediction — always return a valid number, never crash/timeout
2. Proxy-only live market anchoring — Polymarket via Desearch crawl (more reliable in sandbox than direct API)
3. Exact-match passthrough — if Polymarket match quality >= 0.80, pass through directly with minimal adjustment
4. Bounded calibration — tighter clamp [0.05, 0.95], shrink toward empirical base rate 0.43
5. Layered fallbacks — every signal has a fallback, final fallback is BASE_RATE
6. Budget-aware — hard 200s total, abort expensive paths if < 25s left

Key changes from v8.1:
- Removed direct Polymarket API calls (unreliable in sandbox) → Desearch crawl only
- Exact-match passthrough at pm_q >= 0.80 (skip LLM entirely, saves time + improves accuracy)
- Tighter PROB_FLOOR/CEIL (0.05/0.95) to avoid extreme predictions that hurt Brier score
- Removed openrouter call from fast path (too slow, marginal benefit)
- Simplified geo_strikes: anchor-first, LLM only if Indicia has concrete signals
- Single chutes call (no parallel openrouter) unless budget allows
- Fallback chain: PM crawl → OAI → Chutes → BASE_RATE
"""

import asyncio
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx

# ── env ───────────────────────────────────────────────────────────────────────
GATEWAY = os.environ.get("SANDBOX_PROXY_URL", "http://sandbox_proxy").rstrip("/")
RUN_ID  = os.environ.get("RUN_ID") or str(uuid4())

# ── endpoints ─────────────────────────────────────────────────────────────────
PATH_OAI        = GATEWAY + "/api/gateway/openai/responses"
PATH_CHUTES     = GATEWAY + "/api/gateway/chutes/chat/completions"
PATH_CRAWL      = GATEWAY + "/api/gateway/desearch/web/crawl"
PATH_INDICIA    = GATEWAY + "/api/gateway/numinous-indicia"

# ── models ────────────────────────────────────────────────────────────────────
MODEL_OAI_PRIMARY     = "gpt-5-mini"
MODEL_CHUTES_PRIMARY  = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_CHUTES_FALLBACK = "deepseek-ai/DeepSeek-V3-0324"

# ── calibration ───────────────────────────────────────────────────────────────
BASE_RATE        = 0.43
PROB_FLOOR       = 0.05
PROB_CEIL        = 0.95
SHRINK_FRACTION  = 0.08
DEBIAS_LOGIT     = 0.12
EXTREMIZE_D      = 1.15

# ── timing ────────────────────────────────────────────────────────────────────
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time else 0

def _budget_left():
    return max(0, 200 - _elapsed())


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


# ═══════════════════════════════════════════════════════════════════════════════
# Probability parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_prob(txt: str) -> float | None:
    if not txt:
        return None
    for key in ("prediction", "probability", "confidence", "forecast", "prob", "p", "likelihood", "final_probability"):
        hit = re.search(rf'["\']?{key}["\']?\s*:\s*([\d.]+)', txt, re.I)
        if hit:
            v = float(hit.group(1))
            return clamp(v / 100.0 if v > 1.5 else v)
    hit = re.search(r'PREDICTION\s*[:\-]\s*([\d.]+)', txt, re.I)
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
# Segment detection
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_GENERAL = frozenset({"geopolitics", "politics", "iran", "middle east",
                           "elections", "primaries", "global elections"})
_IRAN_KIN_RE = re.compile(
    r"retaliate with.*strike|launch.*rocket|launch.*attack|fire rocket"
    r"|sirens sound|missile strike.*on|strike.*in response"
    r"|drone strike|missile strike|military strike|artillery strike"
    r"|drone attack|air\s?strike|experience.*strike|experience.*attack", re.I)

_SPHERES = (
    ("athletics", frozenset({"MATCH", "GAME", "WIN", "VS", "CHAMPIONSHIP", "LEAGUE", "CUP", "PLAYOFF"})),
    ("governance", frozenset({"ELECTION", "VOTE", "POLL", "PRESIDENT", "GOVERNOR", "SENATOR", "TARIFF", "SANCTION"})),
    ("financial", frozenset({"RATE", "FED", "GDP", "INFLATION", "BITCOIN", "CRYPTO", "PRICE", "MARKET"})),
    ("technology", frozenset({"LAUNCH", "RELEASE", "APP", "SOFTWARE", "UPDATE", "SHIP"})),
    ("showbusiness", frozenset({"MOVIE", "FILM", "OSCAR", "GRAMMY", "EMMY", "ALBUM", "AWARD"})),
)

def classify_segment(record: dict) -> str:
    meta   = record.get("metadata", {})
    topics = {t.lower() for t in meta.get("topics", [])}
    title  = record.get("title", "")
    blob   = (title + " " + record.get("description", "")).lower()
    tl     = title.lower()

    if "strikes" in topics or re.search(
        r"military strike|missile strike|drone (?:attack|strike)|air\s?strike", blob
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
    if "app store" in blob or " app " in blob:
        return "app_store"
    if " temperature " in blob:
        return "weather"
    if "earnings" in blob:
        return "earnings"
    if "inflation" in blob:
        return "inflation"
    if " price of " in blob:
        return "price"
    return "general"


# ═══════════════════════════════════════════════════════════════════════════════
# Geo anchors (unchanged from v8.1 — validated against outcomes)
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_ANCHORS = {
    "ukraine": {
        "sirens":    ("sirens",   0.92),
        "strikes":   ("strikes",  0.85),
        "attack":    ("attack",   0.82),
        "drone":     ("drone",    0.78),
        "default":   ("general",  0.70),
    },
    "israel": {
        "rocket":    ("rocket",   0.75),
        "sirens":    ("sirens",   0.72),
        "strike":    ("strike",   0.68),
        "default":   ("general",  0.60),
    },
    "russia": {
        "attack":    ("attack",   0.55),
        "default":   ("general",  0.45),
    },
    "default": {
        "default":   ("general",  0.40),
    },
}

def geo_anchor(title: str) -> tuple[float, str]:
    tl = title.lower()
    region = "default"
    for r in ("ukraine", "israel", "russia"):
        if r in tl:
            region = r
            break
    anchors = _GEO_ANCHORS[region]
    for key, (qtype, val) in anchors.items():
        if key != "default" and key in tl:
            return val, qtype
    return anchors["default"][1], anchors["default"][0]

def geo_leash(llm_p: float, anchor: float, leash: float = 0.20) -> float:
    return clamp(max(anchor - leash, min(anchor + leash, llm_p)))

def _geo_city(title: str) -> str:
    cities = ["kyiv", "kharkiv", "zaporizhzhia", "dnipro", "odesa", "lviv",
              "kherson", "mykolaiv", "tel aviv", "haifa", "jerusalem", "gaza",
              "moscow", "belgorod"]
    tl = title.lower()
    for city in cities:
        if city in tl:
            return city
    for r in ("ukraine", "israel", "russia"):
        if r in tl:
            return r
    return ""


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
                print(f"[v9] post_retry failed {url}: {exc}")
            await asyncio.sleep(2 ** i)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Polymarket via Desearch crawl (proxy-only — no direct API)
# ═══════════════════════════════════════════════════════════════════════════════

def _pm_match_quality(title: str, pm_question: str) -> float:
    stop = frozenset("will the a an in on of by to and or for at is it from that this with as are was be".split())
    t_words = {w for w in re.findall(r'\w+', title.lower()) if w not in stop and len(w) > 2}
    q_words = {w for w in re.findall(r'\w+', pm_question.lower()) if w not in stop and len(w) > 2}
    if not t_words:
        return 0.0
    return len(t_words & q_words) / len(t_words)

async def polymarket_signal(client, title: str) -> tuple[float | None, float]:
    """Returns (price, match_quality). Uses Desearch crawl only — no direct Polymarket API."""
    if _budget_left() < 30:
        return None, 0.0
    try:
        keywords = " ".join(w for w in title.split() if len(w) > 3)[:80]
        crawl_url = f"https://polymarket.com/markets?_q={keywords.replace(' ', '+')}"
        data = await post_retry(
            client, PATH_CRAWL,
            {"url": crawl_url, "run_id": RUN_ID},
            timeout=30, attempts=2
        )
        if not data:
            return None, 0.0

        text = ""
        if isinstance(data, dict):
            text = data.get("content") or data.get("text") or json.dumps(data)
        elif isinstance(data, str):
            text = data

        # Look for price patterns near question matches
        lines = text.split("\n")
        best_price = None
        best_q = 0.0

        for i, line in enumerate(lines):
            q = _pm_match_quality(title, line)
            if q > 0.35:
                # Look for a price in surrounding lines
                window = " ".join(lines[max(0, i-2):i+3])
                prices = re.findall(r'\b(\d{1,2}(?:\.\d{1,2})?)\s*%|\b(0\.\d{2,3})\b', window)
                for pct, dec in prices:
                    raw = float(pct) / 100.0 if pct else float(dec)
                    if 0.02 < raw < 0.98:
                        if q > best_q:
                            best_price = raw
                            best_q = q

        if best_price is not None:
            print(f"[v9] PM crawl → price={best_price:.3f} q={best_q:.2f}")
        return best_price, best_q

    except Exception as e:
        print(f"[v9] PM crawl error: {e}")
        return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Indicia (geo OSINT)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_indicia(client, city: str, title: str) -> str:
    if not city or _budget_left() < 40:
        return ""
    try:
        r = await client.get(
            PATH_INDICIA,
            params={"location": city, "question": title[:200], "run_id": RUN_ID},
            timeout=20
        )
        r.raise_for_status()
        data = r.json()
        return data.get("summary") or data.get("content") or ""
    except Exception as e:
        print(f"[v9] indicia error: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM signals
# ═══════════════════════════════════════════════════════════════════════════════

_SYS_GENERAL = """\
You are an expert calibrated forecaster. Estimate the probability this event resolves YES.
Use reference-class reasoning and current evidence. Apply base rates.
Respond with ONLY: {"prediction": 0.XX}"""

_SYS_GEO = """\
You are an expert forecaster for military conflict events. Estimate P(YES).
The ANCHOR is your PRIMARY signal — it reflects base rates from hundreds of similar events.
Only deviate from anchor if live data shows CONCRETE evidence for today specifically.
PREDICTION: [0.01-0.99]
REASONING: [2-3 sentences]"""

async def oai_signal(client, question: str, cutoff: str) -> float | None:
    if _budget_left() < 40:
        return None
    prompt = (
        f"Forecasting question: {question}\n"
        f"Resolution cutoff: {cutoff}\n\n"
        "Search for current information, apply reference-class forecasting.\n"
        'Respond with ONLY: {"prediction": 0.XX}'
    )
    data = await post_retry(
        client, PATH_OAI,
        {
            "model": MODEL_OAI_PRIMARY,
            "input": [{"role": "user", "content": prompt}],
            "instructions": _SYS_GENERAL,
            "tools": [{"type": "web_search_preview"}],
            "max_output_tokens": 300,
            "run_id": RUN_ID,
            "reasoning": {"effort": "low"},
        },
        timeout=80, attempts=2,
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
    if _budget_left() < 25:
        return None
    user_msg = (
        f"Question: {question}\n"
        f"Cutoff: {cutoff}\n\n"
        f"Evidence:\n{context[:2500]}\n\n"
        'Respond ONLY with: {"prediction": 0.XX}'
    )
    data = await post_retry(
        client, PATH_CHUTES,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYS_GENERAL},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 100,
            "temperature": 0,
            "stream": False,
            "run_id": RUN_ID,
        },
        timeout=80, attempts=2,
    )
    if not data:
        return None
    txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return parse_prob(txt)


# ═══════════════════════════════════════════════════════════════════════════════
# Segment routers
# ═══════════════════════════════════════════════════════════════════════════════

async def run_geo_strikes(record: dict, client) -> float:
    title  = record.get("title", "")
    cutoff = str(record.get("cutoff", ""))
    anchor, qtype = geo_anchor(title)
    city = _geo_city(title)
    print(f"[v9] geo_strikes anchor={anchor:.2f} qtype={qtype} city={city}")

    # Only call Indicia+LLM if we have budget
    if _budget_left() > 60:
        indicia_data = await fetch_indicia(client, city, title)
        if indicia_data and len(indicia_data) > 50:
            geo_prompt = (
                f"EVENT: {title}\nCutoff: {cutoff}\n\n"
                f"ANCHOR (primary signal): {anchor:.2f} (type={qtype})\n\n"
                f"LIVE DATA:\n{indicia_data[:2000]}"
            )
            data = await post_retry(
                client, PATH_OAI,
                {
                    "model": MODEL_OAI_PRIMARY,
                    "input": [{"role": "user", "content": geo_prompt}],
                    "instructions": _SYS_GEO,
                    "max_output_tokens": 200,
                    "run_id": RUN_ID,
                },
                timeout=50, attempts=2,
            )
            if data:
                for block in data.get("output") or []:
                    if block.get("type") == "message":
                        for part in block.get("content") or []:
                            p = parse_prob(part.get("text", ""))
                            if p is not None:
                                leashed = geo_leash(p, anchor)
                                print(f"[v9] geo LLM={p:.3f} leashed={leashed:.3f}")
                                return leashed

    print(f"[v9] geo_strikes → anchor fallback {anchor:.3f}")
    return anchor


async def run_iran_diplomatic(_record: dict) -> float:
    print("[v9] iran_diplomatic → 0.15")
    return 0.15


async def run_general(record: dict, client) -> float:
    title  = record.get("title", "")
    desc   = record.get("description", "") or ""
    cutoff = str(record.get("cutoff", ""))

    # Step 1: Polymarket via crawl
    pm_price, pm_q = await polymarket_signal(client, title)

    # Exact-match passthrough — skip LLM entirely
    if pm_price is not None and pm_q >= 0.80:
        result = shrink(pm_price, frac=0.03)  # minimal shrink
        print(f"[v9] exact-match passthrough PM={pm_price:.3f} → {result:.3f}")
        return result

    # High-confidence PM fast path
    if pm_price is not None and pm_q >= 0.65 and (pm_price <= 0.06 or pm_price >= 0.94):
        print(f"[v9] fast-path PM={pm_price:.3f} q={pm_q:.2f}")
        return pm_price

    # Step 2: OAI web search
    oai_est = await oai_signal(client, title, cutoff)

    # Build context for Chutes
    ctx_parts = []
    if pm_price is not None:
        ctx_parts.append(f"Polymarket YES price: {pm_price:.3f} (match quality: {pm_q:.2f})")
    if oai_est is not None:
        ctx_parts.append(f"OpenAI web search estimate: {oai_est:.3f}")
    if desc:
        ctx_parts.append(f"Description: {desc[:400]}")
    context = "\n\n".join(ctx_parts) or "No external data available."

    # Step 3: Chutes (only if budget allows)
    chutes_est = await chutes_signal(client, title, context, cutoff)
    if chutes_est is None and _budget_left() > 30:
        chutes_est = await chutes_signal(client, title, context, cutoff, model=MODEL_CHUTES_FALLBACK)

    print(f"[v9] signals — PM:{pm_price} OAI:{oai_est} Chutes:{chutes_est}")

    votes, weights = [], []
    if pm_price is not None:
        votes.append(pm_price);   weights.append(4.0 * pm_q)
    if oai_est is not None:
        votes.append(oai_est);    weights.append(2.5)
    if chutes_est is not None:
        votes.append(chutes_est); weights.append(1.5)

    if not votes:
        print("[v9] no signals → BASE_RATE")
        return BASE_RATE

    result = geo_pool(votes, weights)
    result = debias(result)

    # Only extremize if we have multiple signals
    if len(votes) >= 2:
        result = extremize(result)

    # Shrink toward base rate for single-signal predictions
    if len(votes) == 1:
        result = shrink(result, frac=SHRINK_FRACTION * 2)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def agent_main(event_data: dict) -> dict:
    global _start_time
    _start_time = time.time()

    title   = event_data.get("title", "")
    segment = classify_segment(event_data)

    print(f"[v9] segment={segment} title={title[:80]}")

    # Normalize cutoff
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
            # All other segments go through general flow
            return await run_general(event_data, client)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                fut = pool.submit(asyncio.run, _run())
                prediction = fut.result(timeout=210)
        else:
            prediction = loop.run_until_complete(_run())
    except Exception as e:
        print(f"[v9] FATAL error, returning BASE_RATE: {e}")
        prediction = BASE_RATE

    prediction = clamp(prediction)
    elapsed = _elapsed()
    print(f"[v9] FINAL prediction={prediction:.4f} elapsed={elapsed:.1f}s")

    return {"prediction": prediction}
