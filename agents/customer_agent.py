"""Customer agent for customer lookup and registration."""

from agents.realtime import RealtimeAgent

from ai_service.tools.context import CallContext
from ai_service.tools.customer_tools import CUSTOMER_TOOLS


customer_agent = RealtimeAgent[CallContext](
    name="Customer Agent",
    instructions=(
        "You are a customer service specialist for a salon business. "
        "Help look up existing customer records by phone number, "
        "or register new customers. Collect the caller's name and phone number. "
        "Once you have the customer information, hand off back to the "
        "appropriate agent (Booking or FAQ) to continue."
    ),
    tools=list(CUSTOMER_TOOLS),
)
