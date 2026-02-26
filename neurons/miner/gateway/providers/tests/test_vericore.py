import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.vericore import VericoreClient
from neurons.validator.models.vericore import VericoreResponse

MOCK_VERICORE_RESPONSE = {
    "batch_id": "batch-abc-123",
    "request_id": "req-xyz-456",
    "preview_url": "",
    "evidence_summary": {
        "total_count": 3,
        "neutral": 37.5,
        "entailment": 1.03,
        "contradiction": 61.46,
        "sentiment": -0.07,
        "conviction": 0.82,
        "source_credibility": 0.93,
        "narrative_momentum": 0.48,
        "risk_reward_sentiment": -0.15,
        "political_leaning": 0.0,
        "catalyst_detection": 0.12,
        "statements": [
            {
                "statement": "Evidence supports the claim based on recent data.",
                "url": "https://example.com/article1",
                "contradiction": 0.87,
                "neutral": 0.12,
                "entailment": 0.01,
                "sentiment": -0.5,
                "conviction": 0.75,
                "source_credibility": 0.85,
                "narrative_momentum": 0.5,
                "risk_reward_sentiment": -0.5,
                "political_leaning": 0.0,
                "catalyst_detection": 0.3,
            },
        ],
    },
}


class TestVericoreClient:
    @pytest.fixture
    def client(self):
        return VericoreClient(api_key="test_api_key")

    async def test_calculate_rating_success(self, client: VericoreClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.verify.vericore.ai/calculate-rating/v2",
                status=200,
                body=json.dumps(MOCK_VERICORE_RESPONSE).encode("utf-8"),
            )

            result = await client.calculate_rating(
                statement="Bitcoin will reach $100k by end of 2026"
            )

            assert isinstance(result, VericoreResponse)
            assert result.batch_id == "batch-abc-123"
            assert result.request_id == "req-xyz-456"
            assert result.evidence_summary.total_count == 3
            assert result.evidence_summary.support is None
            assert result.evidence_summary.refute is None
            assert result.evidence_summary.contradiction == 61.46
            assert result.evidence_summary.sentiment == -0.07
            assert len(result.evidence_summary.statements) == 1
            assert result.evidence_summary.statements[0].contradiction == 0.87
            assert result.evidence_summary.statements[0].sentiment == -0.5

    async def test_calculate_rating_with_preview(self, client: VericoreClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.verify.vericore.ai/calculate-rating/v2",
                status=200,
                body=json.dumps(MOCK_VERICORE_RESPONSE).encode("utf-8"),
            )

            result = await client.calculate_rating(
                statement="Test statement", generate_preview=True
            )

            assert isinstance(result, VericoreResponse)
            assert result.batch_id == "batch-abc-123"

    async def test_calculate_rating_server_error(self, client: VericoreClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.verify.vericore.ai/calculate-rating/v2",
                status=500,
                body=b"Internal server error",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.calculate_rating(statement="Test statement")

            assert exc.value.status == 500

    async def test_calculate_rating_authentication_error(self, client: VericoreClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.verify.vericore.ai/calculate-rating/v2",
                status=401,
                body=b"Unauthorized",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.calculate_rating(statement="Test statement")

            assert exc.value.status == 401

    def test_client_initialization_invalid_api_key(self):
        with pytest.raises(ValueError, match="Vericore API key is not set"):
            VericoreClient(api_key="")

        with pytest.raises(ValueError, match="Vericore API key is not set"):
            VericoreClient(api_key=None)
