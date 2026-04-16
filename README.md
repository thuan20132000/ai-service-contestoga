# AI Service

Real-time voice AI receptionist powered by the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) and Twilio. Runs as a FastAPI server alongside the Django API.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service (port 5050)
make run-fastapi

# Or run both Django + FastAPI
make run-both
```

## Architecture

```
ai_service/
  main.py                    # FastAPI app entry point
  config.py                  # Pydantic Settings (env vars)

  agents/
    receptionist.py          # Triage agent with handoffs (per-business)
    faq_agent.py             # FAQ sub-agent (business info, services)
    booking_agent.py         # Booking sub-agent (availability, appointments)
    customer_agent.py        # Customer sub-agent (lookup, registration)

  tools/
    context.py               # CallContext dataclass (per-call state)
    faq_tools.py             # get_business_information, get_service_information
    booking_tools.py         # check_availability, book/look_up/cancel_appointment
    customer_tools.py        # get_customer_information

  handlers/
    twilio_handler.py        # Twilio <-> OpenAI audio bridge

  services/
    business_booking_service.py  # Django ORM booking operations
    call_session_service.py      # CallSession lifecycle management
    openai_api.py                # Chat completions (data sanitization, analysis)
    audio_service.py             # Whisper STT / TTS utilities
    booking_api.py               # Legacy external HTTP API client

  routing/
    main.py                  # Router aggregator
    websocket.py             # WebSocket endpoint (delegates to TwilioHandler)
    twilio.py                # Twilio voice webhooks
    health.py                # Health check endpoints
    booking.py               # Booking test endpoints
```

## Call Flow

```
Incoming Call
  -> Twilio receives call
  -> POST /ai-service/incoming-call
  -> Returns TwiML connecting to WebSocket
  -> WS /ai-service/ws/media-stream/{call_sid}/call_to/{call_to}

WebSocket Handler (routing/websocket.py)
  1. Fetches AIConfiguration for the business (by Twilio phone number)
  2. Creates CallContext with BusinessBookingService
  3. Creates RealtimeAgent with business-specific prompt
  4. Delegates to TwilioHandler

TwilioHandler (handlers/twilio_handler.py)
  - Opens RealtimeSession via OpenAI Agents SDK
  - Runs 3 concurrent async loops:
    - Twilio recv loop: receives g711_ulaw audio, buffers it
    - Buffer flush loop: sends buffered audio to OpenAI every 20ms
    - Realtime event loop: forwards OpenAI audio back to Twilio

Multi-Agent Handoffs
  - Receptionist (triage) determines caller intent
  - Hands off to FAQ Agent, Booking Agent, or Customer Agent
  - Sub-agents can hand off to each other as needed
  - SDK auto-dispatches @function_tool calls per active agent

Call End
  - Twilio sends "stop" event
  - CallSessionService analyzes transcript (outcome, sentiment, summary)
  - CallSession record updated in database
```

## Agents & Tools

### Receptionist (triage)
No tools — routes callers to the right sub-agent via handoffs.

### FAQ Agent (`tools/faq_tools.py`)
| Tool | Description |
|------|-------------|
| `get_business_information` | Business hours, contact, location |
| `get_service_information` | All active salon services |

### Booking Agent (`tools/booking_tools.py`)
| Tool | Description |
|------|-------------|
| `check_availability` | Available time slots for a date/service |
| `book_appointment` | Book an appointment |
| `look_up_appointment` | Find appointments by phone/date |
| `cancel_appointment` | Cancel an appointment |

### Customer Agent (`tools/customer_tools.py`)
| Tool | Description |
|------|-------------|
| `get_customer_information` | Look up or create customer by phone |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8001` | Server port |
| `OPENAI_API_KEY` | required | OpenAI API key |
| `ELEVENLABS_API_KEY` | optional | ElevenLabs API key |

### Per-Business Config (Database)

Each business has an `AIConfiguration` record in Django with:

- `prompt` — system instructions for the AI receptionist
- `model_name` — realtime model (e.g. `gpt-realtime-mini`)
- `voice_provider` — voice ID (e.g. `alloy`)
- `temperature` — model temperature
- `greeting_message` — initial greeting played on answer
- `language` — language code (e.g. `en-US`)

## Adding New Tools

1. Add an async function in the appropriate tools file (`faq_tools.py`, `booking_tools.py`, or `customer_tools.py`):

```python
@function_tool
async def my_new_tool(
    ctx: RunContextWrapper[CallContext],
    param1: str,
    param2: int,
) -> str:
    """Description of what this tool does (shown to the AI).

    Args:
        param1: Description of param1.
        param2: Description of param2.
    """
    result = await ctx.context.booking_service.some_method(param1, param2)
    return json.dumps(result, default=str)
```

2. Add it to the corresponding tools list (`FAQ_TOOLS`, `BOOKING_TOOLS`, or `CUSTOMER_TOOLS`).

3. To add a new sub-agent, create `agents/my_agent.py` with a `RealtimeAgent`, then add a `realtime_handoff()` to it in `agents/receptionist.py`.

The SDK auto-generates the JSON schema from type hints and docstrings. No manual schema definition needed.

## Testing

```bash
# Run AI service tests
make test-fastapi

# Run a specific test
pytest ai_service/tests/test_booking_api.py -v
```

### Manual E2E Testing

1. Expose the service with ngrok: `ngrok http 5050`
2. Configure the Twilio phone number webhook to `https://<ngrok-url>/ai-service/incoming-call`
3. Call the Twilio number
4. Verify: greeting plays, voice conversation works, tool calls execute, call session is saved

## Key Dependencies

- `openai-agents[voice]` — OpenAI Agents SDK with RealtimeAgent support
- `fastapi` / `uvicorn` — HTTP + WebSocket server
- `twilio` — Twilio SDK for call management
- Django ORM (accessed via `asgiref.sync_to_async`) — database operations
