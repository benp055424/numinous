import asyncio
import os
import time
from datetime import datetime

import httpx
from pydantic import BaseModel

RUN_ID = os.getenv("RUN_ID")
if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
VERICORE_URL = f"{PROXY_URL}/api/gateway/vericore"

MAX_RETRIES = 3
BASE_BACKOFF = 1.5

TOTAL_COST = 0.0


class AgentData(BaseModel):
    event_id: str
    title: str
    description: str
    cutoff: datetime
    metadata: dict


async def retry_with_backoff(func, max_retries: int = MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return await func()
        except httpx.TimeoutException as e:
            if attempt < max_retries - 1:
                delay = BASE_BACKOFF ** (attempt + 1)
                print(f"[RETRY] Timeout, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise Exception(f"Max retries exceeded: {e}")
        except httpx.HTTPStatusError as e:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = e.response.text if hasattr(e.response, "text") else str(e)

            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = BASE_BACKOFF ** (attempt + 1)
                    print(f"[RETRY] Rate limited (429), retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Rate limit exceeded: {error_detail}")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception:
            raise


def clip_probability(prediction: float) -> float:
    return max(0.0, min(1.0, prediction))


def derive_prediction(evidence_summary: dict) -> float:
    total = evidence_summary.get("total_count", 0)
    if total == 0:
        return 0.5

    entailment = evidence_summary.get("entailment", 0.0)
    contradiction = evidence_summary.get("contradiction", 0.0)

    # Use entailment vs contradiction as the primary signal
    directional_total = entailment + contradiction
    if directional_total <= 0:
        # No directional evidence -- use sentiment as a weak signal
        sentiment = evidence_summary.get("sentiment", 0.0)
        return clip_probability(0.5 + sentiment * 0.15)

    support_ratio = entailment / directional_total

    # Map support_ratio [0,1] to prediction range [0.1, 0.9] to avoid extremes
    prediction = 0.1 + support_ratio * 0.8

    # Nudge by sentiment (small adjustment, capped at +/-0.1)
    sentiment = evidence_summary.get("sentiment", 0.0)
    prediction += sentiment * 0.1

    return clip_probability(prediction)


async def verify_statement(event: AgentData) -> dict:
    global TOTAL_COST
    print("[VERICORE] Verifying event statement...")

    statement = f"{event.title}. {event.description}"

    async def vericore_call():
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "statement": statement,
                "generate_preview": False,
                "run_id": RUN_ID,
            }
            response = await client.post(
                f"{VERICORE_URL}/calculate-rating",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    result = await retry_with_backoff(vericore_call)

    cost = result.get("cost", 0.01)
    TOTAL_COST += cost

    summary = result["evidence_summary"]
    total = summary["total_count"]

    print(f"[VERICORE] Found {total} evidence sources")
    print(f"[VERICORE] Entailment: {summary.get('entailment', 'N/A')}")
    print(f"[VERICORE] Contradiction: {summary.get('contradiction', 'N/A')}")
    print(f"[VERICORE] Sentiment: {summary.get('sentiment', 'N/A')}")
    print(f"[VERICORE] Conviction: {summary.get('conviction', 'N/A')}")
    print(f"[VERICORE] Source credibility: {summary.get('source_credibility', 'N/A')}")
    print(f"[VERICORE] Cost: ${cost:.4f} | Total: ${TOTAL_COST:.4f}")

    prediction = derive_prediction(summary)
    print(f"[VERICORE] Derived prediction: {prediction:.4f}")

    return {
        "event_id": event.event_id,
        "prediction": prediction,
    }


async def run_agent(event: AgentData) -> dict:
    global TOTAL_COST
    TOTAL_COST = 0.0

    start_time = time.time()

    try:
        result = await verify_statement(event)
    except Exception as e:
        print(f"[AGENT] Error: {e}")
        result = {
            "event_id": event.event_id,
            "prediction": 0.5,
        }

    elapsed = time.time() - start_time
    print(f"[AGENT] Complete in {elapsed:.2f}s")
    print(f"[AGENT] Total run cost: ${TOTAL_COST:.4f}")

    return result


def agent_main(event_data: dict) -> dict:
    event = AgentData.model_validate(event_data)
    print(f"\n[AGENT] Running Vericore verification for event: {event.event_id}")
    print(f"[AGENT] Title: {event.title}")

    return asyncio.run(run_agent(event))
