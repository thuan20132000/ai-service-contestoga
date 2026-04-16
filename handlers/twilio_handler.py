"""Twilio media stream handler bridging Twilio WebSocket with OpenAI Agents SDK RealtimeSession."""

import asyncio
import base64
import json
import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import WebSocket

from agents.realtime import RealtimeAgent, RealtimePlaybackTracker, RealtimeRunner, RealtimeSession

from ai_service.config import settings
from ai_service.services.call_session_service import CallSessionService
from ai_service.tools.context import CallContext
from receptionist.models import AIConfiguration

logger = logging.getLogger(__name__)

# Buffer audio in 20ms chunks before sending to OpenAI
BUFFER_FLUSH_INTERVAL = 0.02


class TwilioHandler:
    """Manages bidirectional audio streaming between Twilio and OpenAI RealtimeSession.

    Uses the OpenAI Agents SDK RealtimeSession for the AI conversation, with
    RealtimePlaybackTracker for accurate interruption handling over telephony.
    """

    def __init__(self, websocket: WebSocket):
        self._twilio_ws = websocket
        self._session: RealtimeSession | None = None
        self._stream_sid: str | None = None
        self._call_sid: str | None = None
        self._call_context: CallContext | None = None
        self._playback_tracker = RealtimePlaybackTracker()
        self._conversation_transcript: list[Dict[str, Any]] = []
        self._audio_buffer = bytearray()

        # Track pending marks: mark_name -> (item_id, content_index, audio_bytes)
        self._mark_counter = 0
        self._pending_marks: dict[str, tuple[str, int, bytes]] = {}

        # Track which item_ids we've already saved to avoid duplicates
        self._saved_item_ids: set[str] = set()

        self._realtime_task: asyncio.Task | None = None
        self._twilio_recv_task: asyncio.Task | None = None
        self._buffer_flush_task: asyncio.Task | None = None
        self._done_event = asyncio.Event()

    async def start(
        self,
        agent: RealtimeAgent,
        ai_config: AIConfiguration,
        call_context: CallContext,
    ) -> None:
        """Start the handler: accept Twilio WS, open RealtimeSession, launch loops."""
        self._call_context = call_context
        self._call_sid = call_context.call_sid

        runner = RealtimeRunner(agent)
        self._session = await runner.run(
            context=call_context,
            model_config={
                "api_key": settings.openai_api_key,
                "initial_model_settings": {
                    "model_name": ai_config.model_name or "gpt-realtime-mini",
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "voice": ai_config.voice or "alloy",
                    "input_audio_transcription": {
                        "model": "gpt-4o-mini-transcribe",
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "interrupt_response": True,
                        "create_response": True,
                    },
                },
                "playback_tracker": self._playback_tracker,
            },
        )

        await self._session.enter()
        await self._twilio_ws.accept()

        self._realtime_task = asyncio.create_task(self._realtime_session_loop())
        self._twilio_recv_task = asyncio.create_task(self._twilio_message_loop())
        self._buffer_flush_task = asyncio.create_task(self._buffer_flush_loop())

        logger.info(f"TwilioHandler started for call {self._call_sid}")

    async def wait_until_done(self) -> None:
        """Block until the call ends."""
        await self._done_event.wait()

    async def cleanup(self) -> None:
        """Cancel tasks and finalize call session."""
        for task in (self._realtime_task, self._twilio_recv_task, self._buffer_flush_task):
            if task and not task.done():
                task.cancel()

        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass

        # Finalize call session in database
        if self._call_sid:
            try:
                service = CallSessionService(self._call_context.openai_service)
                await service.finalize_call(
                    call_sid=self._call_sid,
                    conversation_transcript=self._conversation_transcript,
                )
            except Exception as e:
                logger.error(f"Failed to finalize call {self._call_sid}: {e}")

        logger.info(f"TwilioHandler cleaned up for call {self._call_sid}")

    # ── Twilio → OpenAI ──────────────────────────────────────────────

    async def _twilio_message_loop(self) -> None:
        """Receive messages from Twilio WebSocket and forward audio to OpenAI."""
        try:
            async for raw_message in self._twilio_ws.iter_text():
                msg = json.loads(raw_message)
                event = msg.get("event")

                if event == "connected":
                    logger.info("Twilio media stream connected")

                elif event == "start":
                    self._stream_sid = msg["start"]["streamSid"]
                    logger.info(f"Twilio stream started: {self._stream_sid}")

                elif event == "media":
                    payload = msg["media"]["payload"]
                    audio_bytes = base64.b64decode(payload)
                    self._audio_buffer.extend(audio_bytes)

                elif event == "mark":
                    # Twilio confirms audio up to this mark has been played
                    mark_name = msg.get("mark", {}).get("name", "")
                    mark_info = self._pending_marks.pop(mark_name, None)
                    if mark_info:
                        item_id, content_index, audio_data = mark_info
                        self._playback_tracker.on_play_bytes(
                            item_id, content_index, audio_data
                        )

                elif event == "stop":
                    logger.info("Twilio stream stopped")
                    break

        except Exception as e:
            logger.error(f"Error in Twilio receive loop: {e}")
        finally:
            self._done_event.set()

    async def _buffer_flush_loop(self) -> None:
        """Periodically flush buffered audio to OpenAI RealtimeSession."""
        try:
            while not self._done_event.is_set():
                await asyncio.sleep(BUFFER_FLUSH_INTERVAL)
                if self._audio_buffer and self._session:
                    buffer_data = bytes(self._audio_buffer)
                    self._audio_buffer.clear()
                    await self._session.send_audio(buffer_data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in buffer flush loop: {e}")

    # ── OpenAI → Twilio ──────────────────────────────────────────────

    async def _realtime_session_loop(self) -> None:
        """Receive events from OpenAI RealtimeSession and forward audio to Twilio."""
        try:
            async for event in self._session:
                event_type = event.type

                if event_type == "audio":
                    await self._handle_audio_event(event)

                elif event_type == "audio_interrupted":
                    await self._handle_interruption()

                elif event_type == "audio_end":
                    pass

                elif event_type == "history_updated":
                    await self._handle_history_updated(event)

                elif event_type == "history_added":
                    pass  # handled via history_updated which has transcripts

                elif event_type == "agent_end":
                    logger.info("Agent ended session")

                elif event_type == "error":
                    logger.error(f"Realtime session error: {event.error}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in realtime session loop: {e}")
        finally:
            self._done_event.set()

    async def _handle_audio_event(self, event) -> None:
        """Forward audio from OpenAI to Twilio and track playback via marks."""
        if not self._stream_sid:
            return

        audio_data = event.audio.data
        item_id = event.item_id
        content_index = event.content_index
        base64_audio = base64.b64encode(audio_data).decode("utf-8")

        # Send audio to Twilio
        await self._twilio_ws.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": self._stream_sid,
                    "media": {"payload": base64_audio},
                }
            )
        )

        # Send a mark so Twilio tells us when this chunk was actually played
        self._mark_counter += 1
        mark_name = f"audio_{self._mark_counter}"
        self._pending_marks[mark_name] = (item_id, content_index, audio_data)

        await self._twilio_ws.send_text(
            json.dumps(
                {
                    "event": "mark",
                    "streamSid": self._stream_sid,
                    "mark": {"name": mark_name},
                }
            )
        )

    async def _handle_interruption(self) -> None:
        """Clear Twilio audio buffer when the caller interrupts."""
        if not self._stream_sid:
            return

        await self._twilio_ws.send_text(
            json.dumps({"event": "clear", "streamSid": self._stream_sid})
        )
        # Discard pending marks since unplayed audio was cleared
        self._pending_marks.clear()
        logger.debug("Cleared Twilio audio buffer (interruption)")

    async def _handle_history_updated(self, event) -> None:
        """Process full history to capture transcripts that weren't ready on history_added."""
        timestamp = datetime.now().isoformat()

        for item in event.history:
            item_id = getattr(item, "item_id", None)
            if not item_id or item_id in self._saved_item_ids:
                continue

            if not hasattr(item, "role") or not hasattr(item, "content"):
                continue

            role = item.role
            content_parts = item.content if isinstance(item.content, list) else [item.content]

            for part in content_parts:
                text = None
                if hasattr(part, "transcript") and part.transcript:
                    text = part.transcript
                elif hasattr(part, "text") and part.text:
                    text = part.text

                if text:
                    self._saved_item_ids.add(item_id)
                    speaker = "caller" if role == "user" else "assistant"
                    self._conversation_transcript.append(
                        {
                            "speaker": speaker,
                            "content": text,
                            "timestamp": timestamp,
                        }
                    )
                    logger.debug(f"Transcript [{speaker}]: {text[:80]}...")

                    db_role = "user" if role == "user" else "assistant"
                    await CallSessionService.save_message(
                        call_sid=self._call_sid,
                        role=db_role,
                        content=text,
                    )
