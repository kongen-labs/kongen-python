# Kongen SDK

[![PyPI version](https://img.shields.io/pypi/v/kongenlabs.svg)](https://pypi.org/project/kongenlabs/)
[![Python versions](https://img.shields.io/pypi/pyversions/kongenlabs.svg)](https://pypi.org/project/kongenlabs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python SDK for the [Kongen Labs](https://kongenlabs.life) Pattern Intelligence API -- pattern transfer scoring and LLM reasoning regime detection.

## Install

```bash
pip install kongenlabs
```

## Quick Start

```python
from kongen import KongenClient

client = KongenClient(api_key="kl_live_...")

# Logic: LLM reasoning regime detection (1 KT)
result = client.logic.score("Prove that sqrt(2) is irrational")
print(result.regime, result.confidence_adj)

# Transfer: Pattern scoring (50 KT)
result = client.transfer.score_signal({
    "complexity": 0.7,
    "constraint": 0.3,
    "boundary": 0.8,
    "coherence": 0.6,
    "magnitude": 1.5,
    "balance": 2.33,
    "gradient": 0.5,
})
print(result.classification, result.confidence)
```

## Authentication

Get your API key at [garden.kongenlabs.life](https://garden.kongenlabs.life).

```python
# Pass directly
client = KongenClient(api_key="kl_live_...")

# Or set environment variable
# export KONGEN_API_KEY=kl_live_...
client = KongenClient()
```

## Batch Scoring

Score multiple signals at a discount (40 KT/signal instead of 50):

```python
signals = [
    {"complexity": 0.7, "constraint": 0.3, ...},
    {"complexity": 0.5, "constraint": 0.6, ...},
    {"complexity": 0.9, "constraint": 0.1, ...},
]

results = client.transfer.score_batch(signals)
for r in results:
    print(f"{r.classification}: adj={r.confidence_adj:.3f}")
```

## MCP Integration

To use Kongen with LLM agents over the Model Context Protocol, install the
separate [`kongenlabs-mcp`](https://pypi.org/project/kongenlabs-mcp/) server package:

```bash
pip install kongenlabs-mcp
```

It exposes `score_prompt`, `transfer_score`, `check_usage`, and `route_model`
as MCP tools backed by the same API. See the `kongenlabs-mcp` package
documentation for client configuration.

## Error Handling

```python
from kongen import TokensExhaustedError, APIError

try:
    result = client.logic.score("...")
except TokensExhaustedError:
    print("Out of tokens -- add a payment method")
except APIError as e:
    print(f"API error {e.status_code}: {e.message}")
```

## Token Usage

Every API call consumes Kongen Tokens (KT). Check your balance:

```python
usage = client.token_usage
print(f"{usage.remaining} KT remaining")
```

| Endpoint | Cost |
|----------|------|
| `logic.score()` | 1 KT ($0.0007) |
| `transfer.score_signal()` | 50 KT ($0.035) |
| `transfer.score_batch()` | 40 KT per signal ($0.028) |

## Pricing

**Pay-as-you-go.** Every account gets 1,000 free Kongen Tokens on signup — no credit card required. Above the free allowance, each Kongen Token costs **$0.0007**. No subscriptions, no commitments. Billed monthly based on actual usage.

Enterprise customers can negotiate custom volume pricing -- contact sales@kongenlabs.life.

## Documentation

- [API Documentation](https://kongenlabs.life/docs)
- [Dashboard & API Keys](https://garden.kongenlabs.life)
- [Website](https://kongenlabs.life)
- [Changelog](https://github.com/kongen-labs/kongen-python/blob/main/CHANGELOG.md)

## License

MIT -- see [LICENSE](LICENSE) for details.
