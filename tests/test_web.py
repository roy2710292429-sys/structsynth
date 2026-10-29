import pytest

from structsynth.web import _parse_seeds, _settings_from_payload


def test_web_seed_parser_accepts_strings_and_objects():
    seeds = _parse_seeds(
        [
            "第一道题",
            {"id": "custom", "text": "第二道题"},
            {"text": "第三道题", "type": "demo"},
        ]
    )
    assert [seed.id for seed in seeds] == ["q001", "custom", "q003"]
    assert seeds[2].type == "demo"


def test_web_settings_are_bounded():
    settings = _settings_from_payload(
        {
            "provider": "mock",
            "count": 4,
            "components": 5,
            "clusters": 3,
            "activation_quantile": 0.75,
            "neighbor_fraction": 0.4,
            "temperature": 0.6,
        }
    )
    assert settings.provider == "mock"
    assert settings.count == 4
    assert settings.components == 5
    assert settings.neighbor_fraction == 0.4
    assert settings.temperature == 0.6

    with pytest.raises(ValueError):
        _settings_from_payload({"provider": "mock", "count": 100})
    with pytest.raises(ValueError):
        _settings_from_payload({"provider": "unknown"})


def test_web_settings_support_separate_chat_and_embedding_endpoints():
    settings = _settings_from_payload(
        {
            "provider": "openai",
            "api": {
                "chat_base_url": "https://api.deepseek.com",
                "chat_api_key": "chat-secret",
                "chat_model": "deepseek-flash",
                "embedding_base_url": "https://embedding.example/v1",
                "embedding_api_key": "embedding-secret",
                "embedding_model": "embedding-model",
            },
        }
    )
    assert settings.api is not None
    assert settings.api.base_url == "https://api.deepseek.com"
    assert settings.api.chat_model == "deepseek-flash"
    assert settings.api.embedding_base_url == "https://embedding.example/v1"
    assert settings.api.embedding_api_key == "embedding-secret"
