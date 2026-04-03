"""
Numinous SN6 miner agent v9.

Design goals:
- Maximize reliability (avoid timeout/imputation losses).
- Use proxy-only data access (sandbox-compatible).
- Anchor to live market pricing when high-confidence match exists.
- Keep outputs calibrated and bounded.
"""

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


GATEWAY = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy").rstrip("/")
RUN_ID = os.getenv("RUN_ID") or "local-test-run"

URL_OPENAI_RESPONSES = f"{GATEWAY}/api/gateway/openai/responses"
URL_CHUTES_CHAT = f"{GATEWAY}/api/gateway/chutes/chat/completions"
URL_DESEARCH_CRAWL = f"{GATEWAY}/api/gateway/desearch/web/crawl"
URL_INDICIA = f"{GATEWAY}/api/gateway/numinous-indicia"

POLYMARKET_BASE = "https://gamma-api.polymarket.com"

MODEL_OPENAI = "gpt-5-mini"
MODEL_CHUTES = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_CHUTES_FALLBACK = "openai/gpt-oss-120b"

TIME_BUDGET_SECONDS = 220.0
PRED_MIN = 0.02
PRED_MAX = 0.98
BASE_FALLBACK = 0.41


@dataclass
class MarketMatch:
    question: str
    description: str
    yes_price: float
    similarity: float
    exact: bool


class Budget:
    def __init__(self, seconds: float):
        self.start = time.time()
        self.deadline = self.start + seconds

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.time())

    def has(self, reserve: float) -> bool:
        return self.remaining() > reserve


def clamp(v: float, lo: float = PRED_MIN, hi: float = PRED_MAX) -> float:
    return max(lo, min(hi, float(v)))


def parse_probability(text: str) -> float | None:
    if not text:
        return None

    text = text.strip()

    # JSON-first parsing.
    for candidate in (text, _extract_json_block(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        for key in ("prediction", "probability", "forecast", "p"):
            value = obj.get(key)
            if value is None:
                continue
            try:
                num = float(value)
                return clamp(num / 100.0 if num > 1.2 else num)
            except Exception:
                continue

    # Label-based parsing.
    label_match = re.search(
        r"(prediction|probability|forecast)\s*[:=]\s*([0-9]*\.?[0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if label_match:
        num = float(label_match.group(2))
        return clamp(num / 100.0 if num > 1.2 else num)

    # Percent parsing.
    pct_match = re.search(r"\b([0-9]{1,3}(?:\.[0-9]+)?)\s*%", text)
    if pct_match:
        return clamp(float(pct_match.group(1)) / 100.0)

    # Decimal fallback.
    dec_match = re.search(r"\b(0\.[0-9]{2,4})\b", text)
    if dec_match:
        return clamp(float(dec_match.group(1)))

    return None


def _extract_json_block(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        return fenced.group(1)
    l = text.find("{")
    r = text.rfind("}")
    if l >= 0 and r > l:
        return text[l : r + 1]
    return None


def keyword_query(title: str, max_terms: int = 8) -> str:
    stop = {
        "will",
        "the",
        "a",
        "an",
        "in",
        "on",
        "of",
        "to",
        "and",
        "or",
        "by",
        "for",
        "be",
    }
    words = re.findall(r"[A-Za-z0-9]+", title.lower())
    filtered = [w for w in words if w not in stop and len(w) > 1]
    return " ".join(filtered[:max_terms]) or title[:60]


def jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    sb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def parse_yes_price(market: dict[str, Any]) -> float | None:
    prices = market.get("outcomePrices")
    if prices is not None:
        try:
            arr = json.loads(prices) if isinstance(prices, str) else prices
            if isinstance(arr, list) and arr:
                v = float(arr[0])
                if 0.0 <= v <= 1.0:
                    return v
        except Exception:
            pass

    for token in market.get("tokens") or []:
        if str(token.get("outcome", "")).upper() == "YES":
            try:
                v = float(token.get("price"))
                if 0.0 <= v <= 1.0:
                    return v
            except Exception:
                pass
    return None


def parse_scheduled_date(text: str) -> str | None:
    if not text:
        return None
    iso = re.search(r"scheduled for (\d{4}-\d{2}-\d{2})", text, flags=re.IGNORECASE)
    if iso:
        return iso.group(1)
    natural = re.search(
        r"scheduled for ([A-Za-z]+)\s+([0-9]{1,2})(?:,\s*([0-9]{4}))?",
        text,
        flags=re.IGNORECASE,
    )
    if not natural:
        return None
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = month_map.get(natural.group(1).lower())
    if month is None:
        return None
    day = int(natural.group(2))
    year = int(natural.group(3) or 2026)
    try:
        return datetime(year, month, day).date().isoformat()
    except Exception:
        return None


async def crawl_json(client: httpx.AsyncClient, url: str, budget: Budget) -> dict | None:
    if not budget.has(8):
        return None
    for attempt in range(2):
        try:
            resp = await client.post(
                URL_DESEARCH_CRAWL,
                json={"url": url, "run_id": RUN_ID},
                timeout=min(25.0, max(8.0, budget.remaining() - 1)),
            )
            if resp.status_code in (429, 500, 502, 503) and attempt == 0:
                await asyncio.sleep(1.3)
                continue
            resp.raise_for_status()
            payload = resp.json()
            content = payload.get("content", "")
            if not content:
                return None
            if isinstance(content, str):
                return json.loads(content)
            if isinstance(content, dict):
                return content
            return None
        except Exception:
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            return None
    return None


async def find_polymarket_match(
    client: httpx.AsyncClient, title: str, description: str, budget: Budget
) -> MarketMatch | None:
    query = keyword_query(title)
    url = f"{POLYMARKET_BASE}/public-search?q={query}"
    data = await crawl_json(client, url, budget)
    if not data:
        return None

    event_date = parse_scheduled_date(description)
    best: MarketMatch | None = None
    best_score = -1.0
    for event in data.get("events") or []:
        markets = event.get("markets") or []
        for market in markets:
            q = market.get("question", "")
            sim = jaccard(title, q)
            if sim < 0.25:
                continue
            yes_price = parse_yes_price(market)
            if yes_price is None:
                continue
            desc = market.get("description", "")
            score = sim
            exact = sim >= 0.93

            # Date consistency bump for scheduled events.
            if event_date:
                m_date = parse_scheduled_date(desc)
                if m_date and m_date == event_date:
                    score += 0.08
                elif m_date and m_date != event_date:
                    score -= 0.10

            if score > best_score:
                best_score = score
                best = MarketMatch(
                    question=q,
                    description=desc,
                    yes_price=clamp(yes_price),
                    similarity=sim,
                    exact=exact,
                )
    return best


async def get_indicia_summary(client: httpx.AsyncClient, title: str, budget: Budget) -> str:
    if not budget.has(15):
        return ""
    lower = title.lower()
    if not any(k in lower for k in ("strike", "missile", "drone", "iran", "israel", "ukraine", "war")):
        return ""
    signals: list[str] = []
    for endpoint in ("/x-osint", "/liveuamap"):
        try:
            resp = await client.post(
                URL_INDICIA + endpoint,
                json={"run_id": RUN_ID, "limit": 10},
                timeout=min(20.0, max(6.0, budget.remaining() - 1)),
            )
            resp.raise_for_status()
            for s in (resp.json().get("signals") or [])[:8]:
                msg = str(s.get("signal", "")).strip()
                if msg:
                    signals.append(msg[:180])
        except Exception:
            continue
    if not signals:
        return ""
    return " | ".join(signals[:8])


async def openai_forecast(
    client: httpx.AsyncClient,
    title: str,
    description: str,
    cutoff: str,
    market: MarketMatch | None,
    indicia_summary: str,
    budget: Budget,
) -> float | None:
    if not budget.has(30):
        return None

    market_line = "none"
    if market:
        market_line = f"yes_price={market.yes_price:.3f}, similarity={market.similarity:.2f}"

    prompt = (
        f"Event title: {title}\n"
        f"Event description: {description}\n"
        f"Cutoff: {cutoff}\n"
        f"Market context: {market_line}\n"
        f"Geo OSINT context: {indicia_summary or 'none'}\n\n"
        "Task: estimate probability this resolves YES.\n"
        "Rules:\n"
        "- Use one web search call.\n"
        "- If a reliable market price exists, use it as anchor and adjust only with strong contrary evidence.\n"
        "- Respect literal resolution wording.\n"
        "- Return strict JSON: {\"prediction\": 0.XX, \"reasoning\": \"...\"}\n"
    )

    try:
        resp = await client.post(
            URL_OPENAI_RESPONSES,
            json={
                "model": MODEL_OPENAI,
                "instructions": (
                    "You are a calibrated forecaster. Keep predictions conservative and avoid overconfidence."
                ),
                "tools": [{"type": "web_search"}],
                "input": [{"role": "user", "content": prompt}],
                "reasoning": {"effort": "medium"},
                "run_id": RUN_ID,
            },
            timeout=min(95.0, max(20.0, budget.remaining() - 1)),
        )
        resp.raise_for_status()
        payload = resp.json()
        text_parts: list[str] = []
        for item in payload.get("output") or []:
            if item.get("type") != "message":
                continue
            for c in item.get("content") or []:
                t = c.get("text")
                if t:
                    text_parts.append(t)
        return parse_probability("\n".join(text_parts))
    except Exception:
        return None


async def chutes_fallback(
    client: httpx.AsyncClient,
    title: str,
    description: str,
    cutoff: str,
    market: MarketMatch | None,
    budget: Budget,
) -> float | None:
    if not budget.has(20):
        return None
    market_hint = f"{market.yes_price:.3f}" if market else "none"
    prompt = (
        f"Event: {title}\nDescription: {description}\nCutoff: {cutoff}\n"
        f"Market anchor: {market_hint}\n"
        "Return strict JSON only: {\"prediction\": 0.XX}\n"
    )
    for model in (MODEL_CHUTES, MODEL_CHUTES_FALLBACK):
        try:
            resp = await client.post(
                URL_CHUTES_CHAT,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 160,
                    "run_id": RUN_ID,
                },
                timeout=min(55.0, max(12.0, budget.remaining() - 1)),
            )
            resp.raise_for_status()
            txt = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content", "")
            prob = parse_probability(txt)
            if prob is not None:
                return prob
        except Exception:
            continue
    return None


def domain_of(title: str, desc: str, meta: dict[str, Any]) -> str:
    topics = [str(t).lower() for t in (meta.get("topics") or [])]
    if "weather" in topics or "temperature" in title.lower():
        return "weather"
    if "app store" in " ".join(topics) or "app store" in (title + " " + desc).lower():
        return "app_store"
    if "geopolitics" in topics or any(k in title.lower() for k in ("strike", "war", "missile", "iran", "israel", "ukraine")):
        return "geopolitics"
    if any(k in title.lower() for k in ("bitcoin", "ethereum", "crypto", "price of")):
        return "crypto"
    if "sports" in topics or " vs " in title.lower() or "win?" in title.lower():
        return "sports"
    return "general"


def logit(p: float) -> float:
    p = clamp(p, 0.001, 0.999)
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def post_calibrate(p: float, domain: str) -> float:
    # Slightly positive bias has been associated with current top performers.
    shift = 0.04 if domain in ("general", "geopolitics") else 0.02
    return clamp(inv_logit(logit(p) + shift))


async def forecast(event_data: dict[str, Any]) -> dict[str, Any]:
    budget = Budget(TIME_BUDGET_SECONDS)

    event_id = str(event_data.get("event_id", "unknown"))
    title = str(event_data.get("title", "")).strip()
    description = str(event_data.get("description", "")).strip()
    cutoff = str(event_data.get("cutoff", ""))
    meta = event_data.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    domain = domain_of(title, description, meta)

    # Hard fallback for malformed events.
    if not title:
        return {"event_id": event_id, "prediction": BASE_FALLBACK, "reasoning": "empty_title_fallback"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        market = await find_polymarket_match(client, title, description, budget)
        if market and market.exact and market.similarity >= 0.93:
            pred = post_calibrate(market.yes_price, domain)
            return {
                "event_id": event_id,
                "prediction": pred,
                "reasoning": f"market_exact sim={market.similarity:.2f}",
            }

        indicia = await get_indicia_summary(client, title, budget)
        llm_pred = await openai_forecast(
            client, title, description, cutoff, market, indicia, budget
        )
        if llm_pred is None:
            llm_pred = await chutes_fallback(client, title, description, cutoff, market, budget)

        if llm_pred is None and market is not None:
            pred = market.yes_price
            return {
                "event_id": event_id,
                "prediction": post_calibrate(pred, domain),
                "reasoning": f"market_fallback sim={market.similarity:.2f}",
            }

        if llm_pred is None:
            return {
                "event_id": event_id,
                "prediction": BASE_FALLBACK,
                "reasoning": "model_fallback",
            }

        # Blend with market anchor if available and somewhat relevant.
        pred = llm_pred
        if market is not None:
            if market.similarity >= 0.80:
                w_market = 0.70 if domain in ("sports", "crypto", "app_store") else 0.60
            else:
                w_market = 0.45
            pred = w_market * market.yes_price + (1.0 - w_market) * llm_pred

            # Geo leash around market anchor to reduce overreaction to noisy headlines.
            if domain == "geopolitics":
                low = max(PRED_MIN, market.yes_price - 0.20)
                high = min(PRED_MAX, market.yes_price + 0.20)
                pred = max(low, min(high, pred))

        pred = post_calibrate(pred, domain)
        return {
            "event_id": event_id,
            "prediction": pred,
            "reasoning": f"domain={domain}",
        }


def agent_main(event_data: dict[str, Any]) -> dict[str, Any]:
    try:
        return asyncio.run(forecast(event_data))
    except Exception:
        # Never fail hard: avoid missing prediction penalties.
        return {
            "event_id": str(event_data.get("event_id", "unknown")),
            "prediction": BASE_FALLBACK,
            "reasoning": "fatal_fallback",
        }

