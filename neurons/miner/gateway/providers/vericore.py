import aiohttp

from neurons.validator.models.vericore import VericoreResponse


class VericoreClient:
    __api_key: str
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: dict[str, str]

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Vericore API key is not set")

        self.__api_key = api_key
        self.__base_url = "https://api.verify.vericore.ai"
        self.__timeout = aiohttp.ClientTimeout(total=120)
        self.__headers = {
            "Authorization": f"api-key {self.__api_key}",
            "Content-Type": "application/json",
        }

    async def calculate_rating(
        self, statement: str, generate_preview: bool = False
    ) -> VericoreResponse:
        body = {
            "statement": statement,
            "generate_preview": str(generate_preview).lower(),
        }

        url = f"{self.__base_url}/calculate-rating/v2"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return VericoreResponse.model_validate(data)
