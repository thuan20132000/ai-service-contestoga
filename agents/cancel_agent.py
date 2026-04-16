"""Cancel agent for appointment inquiries (read-only, no direct rescheduling)."""

from agents.realtime import RealtimeAgent

from ai_service.tools.context import CallContext


def create_cancel_agent(caller_number: str) -> RealtimeAgent[CallContext]:
    """Create a cancel agent with the caller's phone number baked into instructions."""
    return RealtimeAgent[CallContext](
        name="Cancel Agent",
        instructions=(
            "You are a receptionist specialist in a salon business. Your role is to help callers cancel their appointments. "
            f"The caller's phone number is: {caller_number}. "
            "Confirm this phone number and name when cancelling the appointment.\n"
            "After getting appointment's date, time and caller's name, let the caller know the appointment will be cancelled and they'll receive confirmation shortly.\n"
            "At the end of the conversation, say politely, naturally, to the caller that you are happy to help and goodbye to the caller.\n"
            "*Always politely, naturally and friendly ask the caller for their name and appointment details before cancelling the appointment.*\n"
        ),
    )
