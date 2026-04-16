"""WebSocket route for Twilio media stream, delegating to TwilioHandler."""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ai_service.agents.receptionist import create_receptionist_agent
from ai_service.handlers.twilio_handler import TwilioHandler
from ai_service.services.business_booking_service import BusinessBookingService
from ai_service.services.call_session_service import CallSessionService
from ai_service.services.openai_service import OpenAIService
from ai_service.tools.context import CallContext
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/media-stream/{call_sid}/call_to/{call_to}")
async def handle_media_stream(
    websocket: WebSocket,
    call_sid: str,
    call_to: str,
):
    """Handle WebSocket connections between Twilio and OpenAI Agents SDK."""
    logger.info(f"========= New media stream connection: call_sid={call_sid}, call_to={call_to}")
    
    # Fetch per-business AI configuration
    ai_config = await CallSessionService.get_ai_configuration(call_to)
    logger.info(f"AI configuration: {ai_config}")

    call_session = await CallSessionService.get_call_session(call_sid)
    logger.info(f"Call session: {call_session}")
    
    caller_number = call_session.caller_number[-10:]
    # Build per-call context
    call_context = CallContext(
        business_id=ai_config.business_id,
        call_sid=call_sid,
        caller_number=caller_number,
        booking_service=BusinessBookingService(ai_config.business_id),
        openai_service=OpenAIService(),
        forward_phone_number=ai_config.forward_phone_number,
    )
    
    logger.info(f"Call context: {call_context}")
    
    # Create agent with business-specific instructions
    agent = create_receptionist_agent(
        instructions=ai_config.prompt, 
        caller_number=caller_number, 
    )
    
    logger.info(f"Agent: {agent}")

    handler = TwilioHandler(websocket)
    logger.info(f"Handler: {handler}")
    try:
        await handler.start(agent, ai_config, call_context)
        await handler.wait_until_done()
        logger.info(f"Call ended: {call_sid}")
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {call_sid}")
    except Exception as e:
        logger.error(f"WebSocket error for {call_sid}: {e}")
        logger.error(f"Exception: {e}")
    finally:
        await handler.cleanup()
        logger.info(f"Handler cleaned up: {call_sid}")
