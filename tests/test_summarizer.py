"""Tests for mempalace.summarizer — wing summary generation via gpt-5.4-mini."""

from unittest.mock import MagicMock, patch

from mempalace import summarizer


# ── summarize_wing ───────────────────────────────────────────────────


def test_summarize_wing_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = summarizer.summarize_wing("justin", ["hello world"])
    assert result == ""


def test_summarize_wing_calls_openai_and_returns_content(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "  A two-sentence summary.  "

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch.object(summarizer, "OpenAI", return_value=fake_client) as mock_ctor:
        result = summarizer.summarize_wing(
            "justin",
            ["I met justin in 2023.", "Justin works on mempalace."],
            max_words=80,
        )

    assert result == "A two-sentence summary."
    mock_ctor.assert_called_once_with(api_key="sk-fake")
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.4-mini"
    assert call_kwargs["max_tokens"] == 200
    assert call_kwargs["temperature"] == 0.3
    prompt = call_kwargs["messages"][0]["content"]
    assert "justin" in prompt
    assert "80 words or fewer" in prompt
    assert "I met justin in 2023." in prompt


def test_summarize_wing_truncates_long_input(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ok"

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    long_chunks = ["x" * 1000] * 20  # 20,000+ chars after joining
    with patch.object(summarizer, "OpenAI", return_value=fake_client):
        summarizer.summarize_wing("big", long_chunks)

    prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "[...truncated]" in prompt
    # Combined section must be capped near the 12k cutoff.
    combined_section = prompt.split("\n\n", 1)[1]
    assert len(combined_section) <= 12_000 + len("\n[...truncated]")


def test_summarize_wing_caps_to_20_samples(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ok"

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    samples = [f"unique_marker_{i}" for i in range(50)]
    with patch.object(summarizer, "OpenAI", return_value=fake_client):
        summarizer.summarize_wing("wing", samples)

    prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    # First 20 must be present, the 21st should not be.
    assert "unique_marker_19" in prompt
    assert "unique_marker_20" not in prompt


def test_summarize_wing_swallows_api_errors(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("boom")

    with patch.object(summarizer, "OpenAI", return_value=fake_client):
        with caplog.at_level("WARNING"):
            result = summarizer.summarize_wing("crashy", ["text"])

    assert result == ""
    assert any("Wing summary failed" in rec.message for rec in caplog.records)
