"""Customer tools for customer lookup and registration."""

import json
from typing import Optional

from agents import function_tool
from agents.run_context import RunContextWrapper

from ai_service.tools.context import CallContext
from ai_service.tools.booking_tools import look_up_appointment

@function_tool
async def get_customer_information(
    ctx: RunContextWrapper[CallContext],
    customer_phone: str,
    customer_name: Optional[str] = None,
) -> str:
    """Get information about a specific customer by phone number.
    Creates a new customer record if not found.

    Args:
        customer_phone: Phone number of the customer.
        customer_name: Name of the customer (used if creating new record).
    """
    data = await ctx.context.booking_service.get_or_create_customer(
        phone_number=customer_phone,
        customer_name=customer_name or "Unknown",
    )
    return json.dumps(data, default=str)


CUSTOMER_TOOLS = [
    get_customer_information,
    look_up_appointment,
]
