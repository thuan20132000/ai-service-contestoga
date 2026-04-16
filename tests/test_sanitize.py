import json

import pytest

from ai_service.services.openai_service import OpenAIService


@pytest.mark.asyncio
async def test_analyze_conversation(monkeypatch):
    """Test that analyze_conversation returns outcome, sentiment, summary, and category."""

    async def mock_generate_response(self, messages):
        return json.dumps(
            {
                "outcome": "successful",
                "sentiment": "positive",
                "summary": "Caller requested a pedicure appointment.",
                "category": "make_appointment",
            }
        )

    monkeypatch.setattr(OpenAIService, "generate_response", mock_generate_response)

    api = OpenAIService()
    conversation = [
        {"role": "user", "content": "I'd like to book a pedicure for tomorrow."},
        {"role": "assistant", "content": "Sure! Let me check availability."},
    ]

    result = await api.analyze_conversation(conversation)
    assert isinstance(result, dict)
    assert result["outcome"] == "successful"
    assert result["category"] == "make_appointment"
