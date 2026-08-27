from binance_algo.logging import redact_sensitive


def test_redacts_nested_secret_fields_and_query_strings() -> None:
    event = {
        "api_key": "public-looking-but-secret",
        "nested": {"signature": "abc", "safe": "value"},
        "url": "https://example.test/path?signature=abc123&symbol=BTCUSDT",
    }

    redacted = redact_sensitive(event)

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["signature"] == "[REDACTED]"
    assert redacted["nested"]["safe"] == "value"
    assert "abc123" not in redacted["url"]
    assert "symbol=BTCUSDT" in redacted["url"]
