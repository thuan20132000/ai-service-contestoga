"""Per-call context passed to all tools via RunContextWrapper."""

from dataclasses import dataclass
from ai_service.services.business_booking_service import BusinessBookingService
from ai_service.services.openai_service import OpenAIService


@dataclass
class CallContext:
    """Context for a single phone call, available to all @function_tool functions."""

    business_id: int
    call_sid: str
    caller_number: str
    booking_service: BusinessBookingService
    openai_service: OpenAIService
    forward_phone_number: str | None = None
