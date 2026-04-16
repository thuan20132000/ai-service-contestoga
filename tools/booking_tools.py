"""Booking tools for appointment management (read-only)."""

import json
import logging

from agents import function_tool
from agents.run_context import RunContextWrapper

from ai_service.tools.context import CallContext

logger = logging.getLogger(__name__)


@function_tool
async def check_availability(
    ctx: RunContextWrapper[CallContext],
    date: str,
    time: str,
    service_ids: list[int],
    service_duration: int,
) -> str:
    """Check availability for a specific service on a given date and time.

    Args:
        date: Date to check availability for (e.g. 2025-01-15).
        time: Time to check availability for (e.g. 10:00 or any).
        service_ids: List of service IDs to check availability for.
        service_duration: Duration of the service in minutes.
    """
    logger.info(f"========== Checking availability: date={date}, time={time}, service_ids={service_ids}, service_duration={service_duration} ==========")
    data = await ctx.context.booking_service.check_availability(
        date=date, 
        time=time, 
        service_ids=service_ids, 
        service_duration=service_duration, 
    )
    return json.dumps(data, default=str)


@function_tool
async def look_up_appointment(
    ctx: RunContextWrapper[CallContext],
    phone_number: str,
    date: str,
) -> str:
    """Get next appointments for a specific client by phone number and date.

    Args:
        phone_number: Phone number of the client (e.g. 0912345678).
        date: Date of the appointment to look up (e.g. 2025-01-01).
    """
    data = await ctx.context.booking_service.lookup_appointments(
        phone_number=phone_number, 
        date=date
    )
    return json.dumps(data, default=str)


@function_tool
async def get_staff_information(
    ctx: RunContextWrapper[CallContext],
    staff_name: str,
) -> str:
    """Get information for a specific staff by staff name."""
    
    data = await ctx.context.booking_service.get_staff_information(staff_name=staff_name)
    return json.dumps(data, default=str)


@function_tool
async def search_services_by_keywords(
    ctx: RunContextWrapper[CallContext],
    keywords: list[str],
) -> str:
    """Search for services by keywords.

    Args:
        keywords: Keywords to search for services by name or description (e.g. ["haircut", "hair styling", "hair coloring", "manicure", "pedicure", "facial", "massage", "etc"]).
    """
    logger.info(f"========== Searching services by keywords: {keywords} ==========")
    data = await ctx.context.booking_service.search_services_by_keywords(keywords=keywords)
    return json.dumps(data, default=str)

BOOKING_TOOLS = [
    check_availability,
    look_up_appointment,
    get_staff_information,
    search_services_by_keywords,
]
