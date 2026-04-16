import json
from typing import Any, Dict, List

import openai

from ai_service.config import settings


class OpenAIService:
    """OpenAI Service for the receptionist."""

    _client: openai.OpenAI
    _model: str
    _temperature: float

    def __init__(self):
        """Initialize the openai api."""
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = "gpt-5-mini"
        self._temperature = settings.openai_temperature

    async def generate_response(self, messages: List[Dict[str, Any]]):
        """Generate a response using the openai api."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        print("Response:: ", response.choices[0].message.content)

        return response.choices[0].message.content

    async def analyze_conversation(self, conversation: List[Dict[str, Any]]) -> str:
        """Analyze a conversation."""
        # conversation = [
        #     {"role": "user", "content": "Hello, how are you?"},
        #     {"role": "assistant", "content": "I'm good, thank you!"},
        # ]
        prompt = f"""
            You have the following conversation:
            {conversation}

            Analyze the conversation and return the outcome, sentiment, summary, and category.

            Return a JSON object containing:
            - outcome: "successful" | "unsuccessful" | "unknown"
            - sentiment: "positive" | "negative" | "neutral"
            - category: determine the caller's primary intent:
              - "make_appointment" if the caller wants to book a new appointment
              - "cancel_appointment" if the caller wants to cancel an existing appointment
              - "reschedule_appointment" if the caller wants to reschedule an existing appointment
              - "ask_question" if the caller is asking general questions (hours, services, pricing, etc.)
            - summary: brief summary of the conversation. If category is make_appointment and client requested specific staff, add the staff name to the summary otherwise leave it as anyone.
            Return only valid JSON.

            Example:
                {{"outcome": "successful", "sentiment": "positive", "summary": "Caller inquired about availability for a pedicure on Friday.", "category": "make_appointment"}}
        """
        messages = [{"role": "user", "content": prompt}]

        response = await self.generate_response(messages)
        response = json.loads(response)
        print("Analyzed conversation response:: ", response)
        return response
