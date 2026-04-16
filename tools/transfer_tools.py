"""Tool for transferring a live call to a human via Twilio REST API."""

import asyncio
import logging

from agents import RunContextWrapper, function_tool
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from main.common_settings import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
from ai_service.tools.context import CallContext

logger = logging.getLogger(__name__)


@function_tool(
    description_override=(
        "Transfer the call to a human manager or staff member. "
        "Use this when the caller explicitly asks to speak with a real person, "
        "a manager, or someone in charge. "
        "IMPORTANT: Before calling this tool, tell the caller you are transferring them now."
    ),
)
async def transfer_to_human(ctx: RunContextWrapper[CallContext]) -> str:
    """Transfer the live Twilio call to the business forward phone number."""
    
    forward_number = ctx.context.forward_phone_number
    if not forward_number:
        return (
            "No forward phone number is configured for this business. "
            "Apologize to the caller and let them know no one is available "
            "to take their call right now. Offer to take a message or help "
            "them with something else."
        )

    call_sid = ctx.context.call_sid
    logger.info(f"Transferring call {call_sid} to {forward_number}")

    try:
      
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        twiml = f"<Response><Dial>{forward_number}</Dial></Response>"
        await asyncio.to_thread(
            client.calls(call_sid).update, twiml=twiml
        )
        logger.info(f"Call {call_sid} transferred to {forward_number}")
        return "Call is being transferred."
    except TwilioRestException as e:
        logger.error(f"Failed to transfer call {call_sid}: {e}")
        return (
            "The transfer failed due to a technical issue. "
            "Apologize to the caller and offer to take a message instead."
        )


TRANSFER_TOOLS = [transfer_to_human]
