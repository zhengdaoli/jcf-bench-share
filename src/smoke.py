"""Make one short request to verify a configured chat provider."""

import argparse
from urllib.parse import urlsplit

import httpx

from .local_config import ConfigError, setting
from .model_adapters import OpenAIAdapter

PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    "gemini": (
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    ),
    "oneapi": ("ONEAPI_KEY", "ONEAPI_BASE_URL", ""),
    "local": (None, "LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    key_name, url_name, default_url = PROVIDERS[args.provider]
    try:
        base_url = setting(url_name, default_url)
        api_key = setting(key_name) if key_name else ""
    except ConfigError as exc:
        parser.error(str(exc))

    try:
        parts = urlsplit(base_url)
        valid_endpoint = (
            parts.scheme in {"http", "https"}
            and parts.hostname
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
        )
    except ValueError:
        valid_endpoint = False
    if not valid_endpoint:
        parser.error(f"Configure {url_name} with a valid endpoint URL.")
    if key_name and not api_key:
        parser.error(f"Configure {key_name} before making a request.")

    client = OpenAIAdapter(args.model, base_url=base_url, api_key=api_key)
    try:
        response = client.generate("Reply with exactly: JCF_OK", max_new_tokens=16)
    except httpx.HTTPStatusError as exc:
        raise SystemExit(f"Provider returned HTTP {exc.response.status_code}.") from None
    except httpx.HTTPError:
        raise SystemExit("Unable to connect to the provider.") from None
    except (ValueError, TypeError, KeyError, IndexError):
        raise SystemExit("Provider returned an unexpected response format.") from None
    if not isinstance(response, str) or response.strip() != "JCF_OK":
        raise SystemExit("The endpoint returned a response without the expected marker.")
    print("Provider smoke test passed.")


if __name__ == "__main__":
    main()
