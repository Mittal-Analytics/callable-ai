from maai import AIModel, get_abs_cost, get_client, get_model_options


def get_model(provider="openrouter", **kwargs) -> AIModel:
    return AIModel(
        name="provider/model",
        api_key="secret",
        provider=provider,
        base_url="https://openrouter.example/v1",
        extra_headers={
            "HTTP-Referer": "https://example.com",
            "X-Title": "Example AI",
        },
        input_tokens_cost_usd=1,
        input_tokens_cached_cost_usd=0.5,
        output_tokens_cost_usd=2,
        output_tokens_reasoning_cost_usd=2,
        **kwargs,
    )


def test_client_uses_model_base_url():
    client = get_client(get_model())

    assert str(client.base_url) == "https://openrouter.example/v1/"
    assert client.api_key == "secret"
    assert "secret" not in repr(get_model())


def test_openrouter_options_use_model_configuration():
    model = get_model(openrouter_providers=["provider/fp8"])

    assert get_model_options(model, "high") == {
        "extra_headers": {
            "HTTP-Referer": "https://example.com",
            "X-Title": "Example AI",
        },
        "extra_body": {
            "transforms": ["middle-out"],
            "provider": {"only": ["provider/fp8"]},
            "reasoning": {"effort": "high", "summary": "concise"},
        },
    }


def test_extra_headers_are_used_for_any_provider():
    model = get_model(provider="compatible-provider")
    options = get_model_options(model, None)

    assert options == {
        "extra_headers": {
            "HTTP-Referer": "https://example.com",
            "X-Title": "Example AI",
        }
    }
    assert options["extra_headers"] is model.extra_headers


def test_cost_uses_model_prices_and_currency_rate():
    model = get_model(usd_to_inr_rate=100)

    assert (
        get_abs_cost(
            {
                "prompt_tokens": 1_000_000,
                "prompt_tokens_details": {"cached_tokens": 500_000},
                "completion_tokens": 1_000_000,
                "completion_tokens_details": {"reasoning_tokens": 250_000},
                "total_tokens": 2_000_000,
            },
            model,
        )
        == 275
    )
