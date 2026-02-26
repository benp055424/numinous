from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

console = Console()

GATEWAY_ENV_PATH = Path("neurons/miner/gateway/.env")

API_KEYS = [
    ("CHUTES_API_KEY", "Chutes", "https://chutes.ai"),
    ("DESEARCH_API_KEY", "Desearch", "https://desearch.ai"),
    ("OPENAI_API_KEY", "OpenAI", "https://platform.openai.com/api-keys"),
    ("PERPLEXITY_API_KEY", "Perplexity", "https://www.perplexity.ai/settings/api"),
    ("VERICORE_API_KEY", "Vericore", "https://vericore.ai"),
]


def _is_key_set(env_content: str, key: str) -> bool:
    if f"{key}=" not in env_content:
        return False
    value = env_content.split(f"{key}=")[1].split("\n")[0].strip()
    return value != ""


def check_env_vars() -> dict[str, bool]:
    if not GATEWAY_ENV_PATH.exists():
        return {key: False for key, _, _ in API_KEYS}

    env_content = GATEWAY_ENV_PATH.read_text()
    return {key: _is_key_set(env_content, key) for key, _, _ in API_KEYS}


def setup_api_keys(force_all: bool = False) -> bool:
    console.print()
    console.print("[cyan]API Key Setup[/cyan]")
    console.print()
    console.print("[dim]You can get your API keys from:[/dim]")
    for _, name, url in API_KEYS:
        console.print(f"[dim]  - {name}: [link={url}]{url}[/link][/dim]")
    console.print()

    env_status = check_env_vars()

    keys_to_set: dict[str, str] = {}

    for env_var, name, _ in API_KEYS:
        if not env_status[env_var] or force_all:
            value = Prompt.ask(
                f"[cyan]{name} API Key[/cyan] [dim](Enter to skip)[/dim]", default=""
            )
            value = value.strip() if value else ""
            if value:
                keys_to_set[env_var] = value

    existing_content = ""
    if GATEWAY_ENV_PATH.exists():
        existing_content = GATEWAY_ENV_PATH.read_text()

    lines = existing_content.split("\n") if existing_content else []

    def update_or_add_key(lines: list[str], key: str, value: str) -> list[str]:
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                updated = True
                break
        if not updated:
            lines.append(f"{key}={value}")
        return lines

    for env_var, value in keys_to_set.items():
        lines = update_or_add_key(lines, env_var, value)

    try:
        GATEWAY_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

        new_content = "\n".join(lines)
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"

        GATEWAY_ENV_PATH.write_text(new_content)

        console.print()
        console.print(f"[green]OK[/green] API keys saved to [cyan]{GATEWAY_ENV_PATH}[/cyan]")
        console.print()

        return True

    except Exception as e:
        console.print()
        console.print(f"[red]Failed to save API keys: {e}[/red]")
        console.print()
        return False
