from dataclasses import dataclass, field

USD_TO_INR_RATE = 110


@dataclass
class AIModel:
    name: str
    api_key: str = field(repr=False, compare=False)

    input_tokens_cost_usd: float
    input_tokens_cached_cost_usd: float
    output_tokens_cost_usd: float
    output_tokens_reasoning_cost_usd: float

    provider: str = "openai"
    base_url: str | None = None
    extra_headers: dict[str, str] | None = field(default=None, repr=False)
    web_search_cost_usd: float | None = None
    lower_token_count_cost: tuple[int, "AIModel"] | None = None
    openrouter_providers: list[str] | None = None
    usd_to_inr_rate: float = USD_TO_INR_RATE

    @property
    def web_search_cost_inr(self) -> float | None:
        return (
            self.usd_to_inr_rate * self.web_search_cost_usd
            if self.web_search_cost_usd is not None
            else None
        )

    @property
    def input_tokens_cost_inr(self) -> float:
        return self.usd_to_inr_rate * self.input_tokens_cost_usd

    @property
    def input_tokens_cached_cost_inr(self) -> float:
        return self.usd_to_inr_rate * self.input_tokens_cached_cost_usd

    @property
    def output_tokens_cost_inr(self) -> float:
        return self.usd_to_inr_rate * self.output_tokens_cost_usd

    @property
    def output_tokens_reasoning_cost_inr(self) -> float:
        return self.usd_to_inr_rate * self.output_tokens_reasoning_cost_usd
