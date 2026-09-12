from unittest.mock import Mock, patch

from app.services import llm


def _ollama_config():
    return {
        "llm_provider": "ollama",
        "ollama_model_name": "qwen3.5:4b",
        "ollama_base_url": "http://127.0.0.1:11434/v1",
    }


def test_cartoon_director_uses_native_ollama_without_thinking():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "message": {
            "content": '{"scenes":[{"scene":1,"layout":"host"}]}'
        }
    }

    with patch("requests.post", return_value=response) as post:
        result = llm.generate_cartoon_direction(
            "Return a JSON cartoon plan.",
            app_config=_ollama_config(),
        )

    assert result == '{"scenes":[{"scene":1,"layout":"host"}]}'
    post.assert_called_once()
    call = post.call_args
    assert call.args[0] == "http://127.0.0.1:11434/api/chat"
    payload = call.kwargs["json"]
    assert payload["model"] == "qwen3.5:4b"
    assert payload["think"] is False
    assert payload["format"] == "json"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.2


def test_cartoon_director_falls_back_when_native_ollama_is_unavailable():
    with (
        patch("requests.post", side_effect=OSError("ollama native endpoint unavailable")),
        patch(
            "app.services.llm._generate_response",
            return_value='{"scenes":[]}',
        ) as fallback,
    ):
        result = llm.generate_cartoon_direction(
            "Return a JSON cartoon plan.",
            app_config=_ollama_config(),
        )

    assert result == '{"scenes":[]}'
    fallback.assert_called_once_with(
        prompt="Return a JSON cartoon plan.",
        app_config=_ollama_config(),
    )
