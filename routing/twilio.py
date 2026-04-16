"""Twilio integration routes for the AI Receptionist application."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from ai_service.config import settings
from ai_service.services import incoming_calling_service

import logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["twilio"])


@router.post("/voice")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls."""
    try:
        # Get form data from Twilio
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        from_number = form_data.get("From")
        to_number = form_data.get("To")

        print(
            f"📞 Incoming call from {from_number} to {to_number}, CallSid: {call_sid}")

        # Create TwiML response to connect to WebSocket
        response = VoiceResponse()
        connect = Connect()

        # Build WebSocket URL
        host = request.url.hostname
        ws_url = f"wss://{host}/ws/twilio-media"
        print(f"🔗 Connecting to WebSocket: {ws_url}")
        stream = Stream(url=ws_url)

        connect.append(stream)
        response.append(connect)

        print(f"🔗 Connecting to WebSocket: {ws_url}")

        return HTMLResponse(content=str(response), media_type="application/xml")

    except Exception as e:
        print(f"❌ Error in Twilio webhook: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(e)}
        )


@router.post("/status")
async def twilio_status_webhook(request: Request):
    """Handle Twilio call status updates."""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        call_duration = form_data.get("CallDuration")

        print(
            f"📊 Call status update - SID: {call_sid}, Status: {call_status}, Duration: {call_duration}")

        return JSONResponse(content={"status": "received"})

    except Exception as e:
        print(f"❌ Error in Twilio status webhook: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )


@router.get("/test")
async def twilio_test_endpoint():
    """Test endpoint for Twilio integration."""
    return {
        "message": "Twilio integration is active",
        "webhook_url": f"{settings.public_ws_url.replace('ws://', 'http://').replace('wss://', 'https://')}/twilio/voice",
        "websocket_url": f"{settings.public_ws_url}/ws/twilio-media",
        "configuration": {
            "auth_token_configured": bool(settings.twilio_media_ws_token),
            "public_ws_url": settings.public_ws_url
        }
    }


@router.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    
    try:
        call_data = await request.form()
        data = dict(call_data)
        
        # Get the phone number of the caller.
        call_from = data.get("From")
        
        # Get the Twilio phone number of the business that the incoming call is to.
        call_to = data.get("To")
        
        # Get the call sid for the incoming call.
        call_sid = data.get("CallSid")
        
        # Get the business for the incoming call
        business = await incoming_calling_service.get_business_by_twilio_number(call_to)

        # AI assistant is enabled: connect to the WebSocket for the AI assistant.
        business_ai_config = await incoming_calling_service.get_active_ai_configuration(business)
        logger.info(f"Business {business.name} has AI assistant enabled: {business.enable_ai_assistant}")

        if business is None:
            logger.warning(f"No business found for incoming call to {call_to}")
            response = VoiceResponse()
            response.say("Sorry, this number is not in service.")
            response.hangup()
            return HTMLResponse(content=str(response), media_type="application/xml")

        # AI assistant is disabled: forward to the business's real phone number if configured.
        logger.info(f"Business {business.name} has AI assistant disabled: {business.enable_ai_assistant}")
        if not business.enable_ai_assistant:
            await incoming_calling_service.create_call_session(
                call_sid=call_sid,
                caller_number=call_from,
                receiver_number=call_to,
                status="forwarded",
                business_id=business.id,
            )

            response = VoiceResponse()
            forward_to = business_ai_config.forward_phone_number or business.phone_number
            if forward_to:
                logger.info(
                    f"Forwarding call {call_sid} for business {business.name} to {forward_to}"
                )
                response.dial(forward_to, caller_id=call_from)
            else:
                logger.warning(
                    f"Business {business.name} has AI assistant disabled and no valid forward number."
                )
                response.say("Sorry, this number is not in service.")
                response.hangup()

            return HTMLResponse(content=str(response), media_type="application/xml")


        if business_ai_config:
            await incoming_calling_service.create_call_session(
                call_sid=call_sid,
                caller_number=call_from,
                receiver_number=call_to,
                status="in_progress",
                business_id=business.id,
            )
            response = VoiceResponse()
            response.say(
                business_ai_config.greeting_message,
                voice="Google.en-US-Chirp3-HD-Aoede",
                language=business_ai_config.language
            )
            host = request.url.hostname
            wss_url = f"wss://{host}/ai-service/ws/media-stream/{call_sid}/call_to/{call_to}"
            connect = Connect()
            connect.stream(url=wss_url)
            response.append(connect)
            return HTMLResponse(content=str(response), media_type="application/xml")


        # No AI configuration found: forward to the business's real phone number if configured.
        await incoming_calling_service.create_call_session(
            call_sid=call_sid,
            caller_number=call_from,
            receiver_number=call_to,
            status="forwarded",
            business_id=business.id,
        )
        response = VoiceResponse()
        if business_ai_config.forward_phone_number:
            logger.info(f"Forwarding call {call_sid} (no AI config) to {business_ai_config.forward_phone_number}")
            response.dial(business_ai_config.forward_phone_number, caller_id=call_from)
        else:
            logger.warning(f"Business {business.name} has no valid forward number; hanging up.")
            response.say("Sorry, this number is not in service.")
            response.hangup()        
        return HTMLResponse(content=str(response), media_type="application/xml")

    except Exception as e:
        logger.error(f"Error in handle_incoming_call: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )