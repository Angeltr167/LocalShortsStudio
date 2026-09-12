from pathlib import Path


path = Path("app/services/llm.py")
text = path.read_text(encoding="utf-8")
old = '''def generate_cartoon_direction(prompt: str, app_config=None) -> str:
    """Generate a bounded cartoon scene plan with the configured text LLM.

    The caller validates every enum and provides a deterministic fallback, so this
    helper intentionally stays thin and never grants the model tool or file access.
    A second attempt helps local Ollama models that occasionally return an empty
    reasoning-only response on their first generation.
    """
    last_response = ""
    for attempt in range(2):
        last_response = (
            _generate_response(prompt=prompt)
            if app_config is None
            else _generate_response(prompt=prompt, app_config=app_config)
        )
        if last_response and not last_response.startswith("Error:"):
            return last_response
        if attempt == 0:
            logger.warning("AI cartoon director returned no usable plan; retrying once")
    return last_response
'''
new = '''def generate_cartoon_direction(prompt: str, app_config=None) -> str:
    """Generate a bounded cartoon scene plan with the configured text LLM.

    For Ollama we prefer its native chat endpoint with ``think=false`` and JSON
    formatting.  Qwen reasoning models can otherwise spend the response budget in a
    thinking channel and leave the OpenAI-compatible ``content`` empty, which is a bad
    fit for a small machine-readable scene plan.  If native Ollama is unavailable or
    rejects the option, the normal provider adapter remains the fallback.
    """
    runtime_app_config = app_config if app_config is not None else config.app
    provider_id = str(
        runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
    ).lower()

    if provider_id == "ollama":
        provider = get_llm_provider("ollama")
        if provider is not None:
            configured_model = runtime_app_config.get(
                provider.config_key("model_name"), ""
            )
            model_name = provider.resolve_model_name(configured_model)
            configured_base_url = runtime_app_config.get(
                provider.config_key("base_url"), ""
            )
            base_url = provider.resolve_base_url(configured_base_url)
            if not base_url:
                base_url = config.get_default_ollama_base_url()
            native_base = re.sub(r"/v1/?$", "", str(base_url).rstrip("/"))
            if native_base and model_name:
                try:
                    import requests

                    response = requests.post(
                        f"{native_base}/api/chat",
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "think": False,
                            "format": "json",
                            "options": {"temperature": 0.2},
                        },
                        timeout=(5, 120),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    message = payload.get("message") if isinstance(payload, dict) else None
                    content = message.get("content") if isinstance(message, dict) else None
                    normalized = str(content or "").strip()
                    if normalized:
                        return normalized
                    logger.warning(
                        "native Ollama cartoon director returned empty content; "
                        "falling back to the configured LLM adapter"
                    )
                except Exception as exc:
                    logger.warning(
                        "native Ollama cartoon director failed; falling back to the "
                        f"configured LLM adapter: {type(exc).__name__}: {exc}"
                    )

    last_response = ""
    for attempt in range(2):
        last_response = (
            _generate_response(prompt=prompt)
            if app_config is None
            else _generate_response(prompt=prompt, app_config=app_config)
        )
        if last_response and not last_response.startswith("Error:"):
            return last_response
        if attempt == 0:
            logger.warning("AI cartoon director returned no usable plan; retrying once")
    return last_response
'''
if text.count(old) != 1:
    raise SystemExit("expected the initial cartoon director helper exactly once")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Ollama cartoon director patch applied")
