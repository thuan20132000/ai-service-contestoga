"""Reschedule agent for appointment inquiries (read-only, no direct rescheduling)."""

from agents.realtime import RealtimeAgent

from ai_service.tools.booking_tools import look_up_appointment
from ai_service.tools.context import CallContext


def create_reschedule_agent(caller_number: str) -> RealtimeAgent[CallContext]:
    """Create a reschedule agent with the caller's phone number baked into instructions."""
    return RealtimeAgent[CallContext](
        name="Reschedule Agent",
        instructions=(
            "You are a reschedule specialist for a salon business. "
            "Help callers reschedule their appointments.\n\n"
            f"The caller's phone number is: {caller_number}. "
            "Confirm this phone number when looking up appointments.\n\n"
            "Use the look_up_appointment tool with this phone number and ask the caller for the date.\n"
            "If the appointment is found, confirm the details and ask for the new preferred date/time. "
            "Then let the caller know the appointment will be rescheduled and they'll receive confirmation shortly.\n"
            "If no appointment is found, offer to help book a new one and transfer to the booking agent.\n"
            "At the end of the conversation, say politely, naturally, to the caller that you are happy to help and goodbye to the caller."
        ),
        tools=[look_up_appointment],
    )
