from structsynth.experiment import APISettings, ExperimentSettings, build_llm
from structsynth.llm import OpenAICompatibleLLM


def test_client_keeps_chat_and_embedding_credentials_separate():
    client = build_llm(
        ExperimentSettings(
            provider="openai",
            api=APISettings(
                base_url="https://api.deepseek.com",
                api_key="chat-key",
                chat_model="deepseek-flash",
                embedding_base_url="https://embed.example/v1",
                embedding_api_key="embed-key",
                embedding_model="embed-model",
            ),
        )
    )
    assert isinstance(client, OpenAICompatibleLLM)
    assert client.chat_base_url == "https://api.deepseek.com"
    assert client.chat_api_key == "chat-key"
    assert client.embedding_base_url == "https://embed.example/v1"
    assert client.embedding_api_key == "embed-key"
