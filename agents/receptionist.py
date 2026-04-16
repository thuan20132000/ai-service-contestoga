"""AI Receptionist — triage agent that hands off to specialized sub-agents."""

from agents.realtime import RealtimeAgent, realtime_handoff

from ai_service.agents.booking_agent import create_booking_agent
from ai_service.agents.customer_agent import customer_agent
from ai_service.agents.faq_agent import faq_agent
from ai_service.agents.reschedule_agent import create_reschedule_agent
from ai_service.agents.cancel_agent import create_cancel_agent
from ai_service.tools.context import CallContext
from ai_service.tools.transfer_tools import TRANSFER_TOOLS


def create_receptionist_agent(instructions: str, caller_number: str) -> RealtimeAgent[CallContext]:
    """Create the triage receptionist agent with handoffs to sub-agents.

    The receptionist greets the caller, determines intent, and hands off to:
    - FAQ Agent: business hours, location, services info
    - Booking Agent: check availability, look up appointments, collect booking details
    - Customer Agent: customer lookup, registration
    - Reschedule Agent: appointment rescheduling
    - Cancel Agent: appointment cancellation

    Sub-agents can also hand off to each other as needed.

    Args:
        instructions: System prompt from AIConfiguration.prompt for this business.
        caller_number: Phone number of the caller.
    """
    # Create reschedule agent with caller's phone number baked in
    reschedule_agent = create_reschedule_agent(caller_number)

    # Create booking agent with caller's phone number baked in
    booking_agent = create_booking_agent(caller_number)


    customer_agent.handoffs = [
        realtime_handoff(
            faq_agent,
            tool_description_override="Transfer to the FAQ Agent for business or service questions.",
        ),
    ]

    reschedule_agent.handoffs = [
        realtime_handoff(
            faq_agent,
            tool_description_override="Transfer to the FAQ Agent for business or service questions.",
        ),
    ]
    
    return RealtimeAgent[CallContext](
        name="AI Receptionist",
        instructions=instructions,
        tools=TRANSFER_TOOLS,
        handoffs=[
            realtime_handoff(
                faq_agent,
                tool_description_override="Transfer to the FAQ Agent for business hours, location, or service details or appointment lookup.",
            ),
            realtime_handoff(
                reschedule_agent,
                tool_description_override="Transfer to the Reschedule Agent for appointment rescheduling.",
            ),
            realtime_handoff(
                booking_agent,
                tool_description_override="Transfer to the Booking Agent for appointment booking and service details.",
            ),
        ],
    )
