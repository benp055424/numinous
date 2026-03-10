import aiohttp

from neurons.validator.models.numinous_indicia import IndiciaSignal

DEFAULT_BASE_URL = "https://indicia.numinouslabs.io"
DEFAULT_TIMEOUT = 30.0


class NuminousIndiciaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def x_osint(self, account: str | None = None, limit: int = 20) -> list[IndiciaSignal]:
        params: dict[str, str | int] = {"limit": limit}
        if account:
            params["account"] = account
        return await self._get_signals("/signals/x_osint", params)

    async def liveuamap(self, region: str | None = None, limit: int = 50) -> list[IndiciaSignal]:
        params: dict[str, str | int] = {"limit": limit}
        if region:
            params["region"] = region
        return await self._get_signals("/signals/liveuamap", params)

    async def _get_signals(self, path: str, params: dict[str, str | int]) -> list[IndiciaSignal]:
        url = f"{self.base_url}{path}"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return [IndiciaSignal(**item) for item in data]
