"""Service for managing call sessions and system logs."""

import logging
from typing import Any, Dict, Optional

from asgiref.sync import sync_to_async
from django.utils import timezone

from ai_service.services.openai_service import OpenAIService
from client.models import Client
from notifications.models import Notification
from main.utils import get_business_managers_group_name
from notifications.services import NotificationDispatcher, NotificationService
from receptionist.models import (AIConfiguration, AIConfigurationStatus,
                                 CallSession, ConversationMessage, SystemLog)

logger = logging.getLogger(__name__)


class CallSessionService:
    """Manages CallSession lifecycle and system logging."""

    def __init__(self, openai_service: Optional[OpenAIService] = None):
        self._openai_service = openai_service or OpenAIService()
        
    @staticmethod
    async def get_call_session(call_sid: str) -> CallSession:
        """Fetch the call session for a call sid."""
        return await CallSession.objects.aget(call_sid=call_sid)

    @staticmethod
    async def get_ai_configuration(call_to: str) -> AIConfiguration:
        """Fetch AIConfiguration for a business by its Twilio phone number."""
        return await AIConfiguration.objects.filter(
            business__twilio_phone_number=call_to,
            status=AIConfigurationStatus.ACTIVE.value,
        ).afirst()

    @staticmethod
    async def get_business_client(call_sid: str) -> Client:
        """Fetch the business client for a call session."""
        call_session = await CallSession.objects.aget(call_sid=call_sid)
        caller = call_session.caller_number

        # (e.g., "+12894428808" -> "2894428808")
        caller = caller[-10:]

        return await Client.objects.filter(
            primary_business_id=call_session.business_id,
            phone=caller,
            is_active=True,
            is_deleted=False,
        ).afirst()

    async def finalize_call(
        self,
        call_sid: str,
        conversation_transcript: list[Dict[str, Any]],
    ) -> None:
        """Analyze conversation and update CallSession on call completion."""
        logger.info(f"Finalizing call session: {call_sid}")

        update_kwargs: Dict[str, Any] = {
            "status": "completed",
            "ended_at": timezone.now(),
            "conversation_transcript": conversation_transcript,
        }

        if conversation_transcript:
            try:
                outcome = await self._openai_service.analyze_conversation(
                    conversation_transcript
                )
                update_kwargs["outcome"] = outcome.get("outcome", "unknown")
                update_kwargs["sentiment"] = outcome.get("sentiment", "neutral")
                update_kwargs["transcript_summary"] = outcome.get("summary", "Unknown")
                update_kwargs["category"] = outcome.get("category", "unknown")
            except Exception as e:
                logger.error(f"Failed to analyze conversation for {call_sid}: {e}")

        await CallSession.objects.filter(call_sid=call_sid).aupdate(**update_kwargs)

        await self._notify_manager(call_sid, update_kwargs)

    @staticmethod
    async def save_message(
        call_sid: str,
        role: str,
        content: str,
    ) -> None:
        """Save a conversation message to the database.

        Args:
            call_sid: The call session ID.
            role: Message role — 'user' or 'assistant'.
            content: The message text.
        """
        try:
            call = await CallSession.objects.aget(call_sid=call_sid)
            await ConversationMessage.objects.acreate(
                call=call,
                role=role,
                content=content,
            )
        except CallSession.DoesNotExist:
            logger.warning(f"Cannot save message: CallSession {call_sid} not found")
        except Exception as e:
            logger.error(f"Failed to save conversation message for {call_sid}: {e}")

    async def _notify_manager(
        self,
        call_sid: str,
        call_data: Dict[str, Any],
    ) -> None:
        """Send push notification to business managers after call categorization."""
        try:
            call_session = await CallSession.objects.select_related("business").aget(
                call_sid=call_sid
            )
            if not call_session.business:
                return

            category = call_data.get("category", "unknown")
            summary = call_data.get("transcript_summary", "")
            caller = call_session.caller_number

            category_labels = {
                "make_appointment": "New Appointment Request",
                "cancel_appointment": "Cancellation Request",
                "reschedule_appointment": "Reschedule Request",
                "ask_question": "General Inquiry",
                "unknown": "Call Completed",
            }
            title = category_labels.get(category, "Call Completed")
            body = f"Caller {caller}: {summary}"

            NotificationDispatcher().dispatchAsync(
                title=title,
                body=body,
                business_id=call_session.business_id,
                channel=Notification.Channel.PUSH,
                group_name=get_business_managers_group_name(call_session.business_id),
                to=None,
                data={
                    "call_sid": call_sid,
                    "caller": caller,
                    "summary": summary,
                    "category": category,
                    "business_id": call_session.business_id,
                },
            )
            
        except Exception as e:
            logger.error(f"Failed to notify manager for call {call_sid}: {e}")

    @staticmethod
    async def create_system_log(
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        call: Optional[CallSession] = None,
    ) -> None:
        """Create a system log entry."""
        await SystemLog.objects.acreate(
            level=level,
            call=call,
            message=message,
            metadata=metadata or {},
        )
