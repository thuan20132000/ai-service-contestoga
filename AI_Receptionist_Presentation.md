# GROUP 5 - CSCN8010 Final Project

## - Minh Thuan (8730956)
## - Preeja Anilal (8791796)
## - Anthony Nosa Izevbokun (9016626)

### https://github.com/thuan20132000/ai-service-contestoga.git

# 🤖 Smart AI Receptionist
### Real-Time Voice Intelligence Powered by OpenAI Agents SDK & Twilio

---

## Executive Summary

The **Smart AI Receptionist** is a production-grade, real-time voice AI system that answers incoming phone calls, understands caller intent, detects emotional distress, and either resolves queries autonomously or escalates to a human agent. It is built on a dual-server architecture — a **FastAPI voice layer** for real-time telephony and a **Django API** as the business logic backbone — bridged by the **OpenAI Agents SDK**.

> **The system doesn't just talk. It listens, understands, classifies, and decides.**

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Deliverable 1 — NLP Model for Query Processing](#deliverable-1--nlp-model-for-query-processing)
3. [Deliverable 2 — LLM Backbone with Deep Learning Classifier Head](#deliverable-2--llm-backbone-with-deep-learning-classifier-head)
4. [Deliverable 3 — Distress Detection & Escalation Model](#deliverable-3--distress-detection--escalation-model)
5. [Deliverable 4 — Response Generator & Off-Ramp Logic](#deliverable-4--response-generator--off-ramp-logic)
6. [Technology Stack](#technology-stack)
7. [Data Flow: End-to-End Call Lifecycle](#data-flow-end-to-end-call-lifecycle)
8. [Deployment & Infrastructure](#deployment--infrastructure)

---

## System Architecture Overview

```
Caller (Phone)
     │
     ▼
[ Twilio PSTN / Voice ]
     │  (audio stream via WebSocket)
     ▼
[ FastAPI Server ]  ←── Real-time voice layer
     │  (REST API calls)
     ▼
[ OpenAI Agents SDK ]
     │  (orchestration + tool calls)
     ▼
[ Django API ]  ←── Business logic, database, agent tools
     │
     ▼
[ PostgreSQL / Data Store ]
```

![System Architecture Overview](./Architecture.png)


The system operates as two cooperating servers:

- **FastAPI** handles the low-latency WebSocket connection from Twilio, streams audio to OpenAI's Realtime API, and manages session state.
- **Django** acts as the authoritative backend: it stores conversation history, exposes tool endpoints that the agent calls (e.g., `lookup_appointment`, `create_ticket`), and manages escalation workflows.

---

## Deliverable 1 — NLP Model for Query Processing

### What It Does

The NLP layer is the **first point of intelligence** in the pipeline. The moment a caller's speech is transcribed (via OpenAI Whisper / Realtime API), the text is passed through a natural language processing pipeline to extract structured meaning before any decision-making occurs.

### Core Components

**Intent Recognition**
The NLP model identifies *what the caller wants* — their primary intent. Examples:

- `general_question` — "What are your hours?"


**Named Entity Recognition (NER)**
Key entities are extracted from the utterance to parameterize the response:
- **Person names** — "My name is John Smith"
- **Dates & times** — "Next Tuesday at 3pm"
- **Account or reference numbers** — "My order number is 84521"
- **Locations** — "The downtown branch"

**Slot Filling & Context Tracking**
Multi-turn conversations require the model to maintain a *dialogue state* — remembering what has already been said so it can ask targeted follow-up questions rather than re-asking for information already provided.

### Implementation Approach

```python
# Simplified NLP pipeline within the agent tool call
def process_query(transcript: str, context: dict) -> dict:
    intent = classify_intent(transcript)          # Intent classification
    entities = extract_entities(transcript)        # SpaCy / OpenAI function calling
    slots = fill_slots(intent, entities, context)  # Dialogue state manager
    return {"intent": intent, "entities": entities, "slots": slots}
```

The NLP layer is embedded directly into the **OpenAI Agents SDK tool definitions**, where function-calling acts as the structured extraction mechanism — the LLM both understands and formats output simultaneously.

An handler is used to collect the NER entities and saved to our database. see code below 

```python

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
```

---

## Deliverable 2 — LLM Backbone with Deep Learning Classifier Head

### What It Does

The core reasoning engine is **GPT-4o** (via the OpenAI Agents SDK), augmented with a **classifier head** — a specialized classification layer that takes the LLM's contextual understanding and maps it to discrete decision categories.

### The LLM Backbone

The **OpenAI Agents SDK** provides the orchestration framework:

- **Agents** are LLM instances configured with system instructions, available tools, and handoff targets.
- **Tools** are Python functions the agent can invoke — these call the Django API endpoints.
- **Handoffs** allow one agent to transfer the conversation to a specialized sub-agent (e.g., a billing specialist agent, an appointment agent).

```python
# Agent definition using the OpenAI Agents SDK
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

```

### The Deep Learning Classifier Head

On top of the LLM's raw output, a **classifier head** produces probabilistic category scores. This is a lightweight neural network layer that:

- Takes the LLM's hidden state / embedding as input
- Outputs a **probability distribution** over action categories
- Makes the decision pipeline **auditable and deterministic** rather than solely prompt-dependent

**Classification Categories:**

| Category | Description | Action Triggered |
|---|---|---|
| `SELF_SERVE` | Query can be resolved by the AI | Continue AI conversation |
| `SOFT_HANDOFF` | Query is complex but not urgent | Queue for callback |
| `HARD_HANDOFF` | Caller requests human or is distressed | Live transfer to agent |
| `DISTRESS` | Emotional distress detected | Emergency escalation pathway |
| `SPAM / ROBOCALL` | Automated caller detected | Graceful termination |

The classifier head enables **confidence scoring** — if the model is uncertain (e.g., confidence < 0.7), it defaults to asking a clarifying question rather than acting on a low-confidence classification.

---

## Deliverable 3 — Distress Detection & Escalation Model(this is not impleted yet on our project)

### What It Does

The distress detection model is a **parallel, always-running analysis layer** that monitors every caller utterance for signals of emotional distress, urgency, or vulnerability — independent of intent classification. When triggered, it overrides the standard response pathway and activates an escalation protocol.

### Signal Detection Layers

**Lexical Signals — What words are used**
- High-urgency vocabulary: "emergency," "crisis," "can't breathe," "help me"
- Frustration markers: "this is unacceptable," "I've called ten times," "nobody helps"
- Vulnerability language: "I don't know what to do," "I'm at my wit's end"

**Acoustic / Prosodic Signals — How it's said**
Via the OpenAI Realtime API's audio stream analysis:
- Speech rate acceleration (talking faster than baseline)
- Elevated pitch / vocal tension
- Irregular pauses or voice breaks
- Crying or heavy breathing detected in audio stream

**Sentiment & Emotional State Tracking**
A rolling sentiment window tracks emotional trajectory across the conversation:

```
Turn 1:  Sentiment = Neutral (0.5)
Turn 2:  Sentiment = Mildly Frustrated (0.35)
Turn 3:  Sentiment = Distressed (0.15)  → ESCALATION THRESHOLD CROSSED
```

### Escalation Decision Matrix

| Distress Score | Sentiment Score | Wait Time | Action |
|---|---|---|---|
| < 0.3 | > 0.5 | Any | Normal AI response |
| 0.3 – 0.6 | 0.3 – 0.5 | < 5 min | Empathetic response + offer human |
| > 0.6 | < 0.3 | Any | **Hard escalation: live transfer** |
| > 0.8 | Any | Any | **Emergency escalation: priority queue** |

### Escalation Activation

```python
async def check_distress(transcript: str, audio_features: dict, history: list) -> EscalationDecision:
    distress_score = distress_model.predict(transcript, audio_features)
    sentiment_trajectory = sentiment_tracker.get_trajectory(history)
    
    if distress_score > HARD_ESCALATION_THRESHOLD:
        await trigger_live_transfer(priority="HIGH")
        return EscalationDecision(action="HARD_HANDOFF", reason="distress_detected")
    elif distress_score > SOFT_ESCALATION_THRESHOLD:
        return EscalationDecision(action="OFFER_HUMAN", reason="elevated_distress")
    
    return EscalationDecision(action="CONTINUE", reason="within_normal_range")
```

This model runs **asynchronously and in parallel** with the main response generation — it never adds latency to the conversation but can interrupt the response pipeline at any point.

---

## Deliverable 4 — Response Generator & Off-Ramp Logic(this is not yet implemented on our project)

### What It Does

The response layer is where the AI formulates its reply and decides which **off-ramp pathway** to execute — whether to continue the conversation, transfer to a human, schedule a callback, or gracefully end the call.

### Response Generation

The response generator uses the **OpenAI Realtime API** for sub-200ms, streaming text-to-speech output, ensuring natural conversational pacing. Responses are shaped by:

- **Tone calibration** — Professional, warm, and empathetic; adjusted dynamically based on distress signals
- **Template anchoring** — Core information (hours, policies, procedures) is retrieved from Django via tool calls, ensuring factual accuracy
- **Persona consistency** — The agent maintains a consistent name, personality, and communication style across the entire call

```python
# Response shaping based on context
def shape_response(intent_result, distress_level, tool_output):
    tone = "empathetic" if distress_level > 0.4 else "professional"
    return ResponseConfig(
        tone=tone,
        max_tokens=150,           # Keep responses concise for voice
        include_confirmation=True, # Repeat key info back to caller
        offer_followup=True
    )
```

### Off-Ramp Logic

Off-ramps are the **exit pathways** out of the AI conversation. Each off-ramp is a deliberate, structured transition with full context preservation so the receiving party (human agent, callback system, CRM) inherits the complete conversation state.

**Off-Ramp 1: Self-Resolution**
The most common pathway. The AI fully handles the request and ends the call gracefully.
```
Caller: "What are your opening hours?"
AI: Retrieves hours via tool call → Answers → Confirms → "Is there anything else I can help you with?" → Call concluded
```

**Off-Ramp 2: Warm Transfer (Soft Handoff)**
The AI has partially resolved the query but determines a human should complete it. The AI briefs the human agent with a summary before connecting.
```
Trigger conditions: Caller explicitly requests human | Complex billing dispute | Unanswered after 2 clarification attempts
Action: AI says "I'm going to connect you with one of our specialists — I'll let them know what we've discussed" → Twilio conference join → Summary pushed to agent CRM screen
```

**Off-Ramp 3: Scheduled Callback**
When wait times are long or caller prefers not to wait. The AI collects callback details and creates a task in Django.
```
Action: Collect name, phone, preferred time → POST /api/callbacks/ in Django → Confirm back to caller → Graceful end
```

**Off-Ramp 4: Emergency Escalation**
Triggered exclusively by the distress detection model. Bypasses all queues.
```
Action: Immediate priority transfer → Alert pushed to supervisor → Full transcript + audio clip flagged for review → Compliance log entry created
```

**Off-Ramp 5: Graceful Deflection**
For out-of-scope requests (legal, medical advice, competitor comparisons). The AI declines politely and redirects.
```
Trigger: Out-of-scope intent detected with confidence > 0.85
Action: "That's not something I'm able to help with, but here's how you can reach the right resource..." → Provide contact info → End call
```

### Off-Ramp State Machine

```
                    ┌─────────────────┐
                    │   Conversation  │
                    │     Active      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
      Resolved?        Distress?      Unresolvable?
              │              │              │
              ▼              ▼              ▼
     Self-Resolution   Emergency      Warm Transfer
          (End)       Escalation      or Deflection
                         (Transfer)      (Redirect)
                                │
                         Callback
                         Offered?
                                │
                         Scheduled
                         Callback
                           (End)
```

---

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Telephony** | Twilio Voice + ConversationRelay | PSTN connection, audio streaming |
| **Voice Layer Server** | FastAPI + Uvicorn | Real-time WebSocket management |
| **AI Orchestration** | OpenAI Agents SDK | Agent definitions, tool calls, handoffs |
| **LLM / Voice AI** | GPT-4o Realtime API | Speech understanding + response generation |
| **Business Logic API** | Django REST Framework | Data access, CRM integration, tool endpoints |
| **Database** | PostgreSQL | Conversation logs, escalation records, callbacks |
| **NLP Utilities** | SpaCy / HuggingFace Transformers | Entity extraction, intent classification support |
| **Distress Analysis** | Custom PyTorch classifier | Sentiment + distress scoring |
| **Deployment** | Docker + Nginx | Containerised dual-server deployment |

---

## Data Flow: End-to-End Call Lifecycle

```
1. INBOUND CALL
   Caller dials → Twilio receives → Webhook fires → FastAPI /incoming-call

2. SESSION INITIATION
   FastAPI opens WebSocket to OpenAI Realtime API
   Agent SDK loads receptionist_agent with tools and context

3. AUDIO STREAMING
   Twilio streams μ-law audio → FastAPI → OpenAI Realtime API
   OpenAI transcribes + processes in real-time

4. NLP PROCESSING
   Transcript → Intent classifier → Entity extractor → Slot filler
   Parallel: Distress detector monitors every utterance

5. TOOL EXECUTION
   Agent decides to call a tool → FastAPI relays → Django API
   e.g., GET /api/appointments/?phone=+16135550101
   Django queries DB → Returns structured result → Agent uses in response

6. RESPONSE SYNTHESIS
   GPT-4o generates response → Text-to-speech → Audio stream back to Twilio → Caller hears response

7. OFF-RAMP EVALUATION
   After each turn: Is query resolved? | Distress threshold crossed? | Escalation requested?
   → Execute appropriate off-ramp pathway

8. CALL CONCLUSION
   Full transcript + metadata → Saved to Django → Compliance log created
```

---

## Deployment & Infrastructure

The system is containerised using **Docker Compose** with two primary services:

```yaml
services:
  fastapi:
    # Real-time voice layer
    # Exposed on port 8000
    # Requires low-latency networking to Twilio + OpenAI

  django:
    # Business logic API
    # Exposed on port 8080 (internal)
    # Connects to PostgreSQL

  postgres:
    # Persistent data store
```

**Environment Configuration:**

```
OPENAI_API_KEY          → GPT-4o Realtime API access
TWILIO_ACCOUNT_SID      → Twilio authentication
TWILIO_AUTH_TOKEN       → Twilio authentication
TWILIO_PHONE_NUMBER     → Inbound call routing
DJANGO_SECRET_KEY       → Django application security
DATABASE_URL            → PostgreSQL connection
NGROK_URL / DOMAIN      → Public WebSocket endpoint for Twilio
```

---

## Key Differentiators

- **Sub-200ms voice latency** via the OpenAI Realtime API — conversations feel natural, not robotic
- **Parallel distress detection** — runs asynchronously so it never adds delay to normal interactions
- **Stateful multi-turn conversations** — context is maintained across all turns within a call session
- **Full audit trail** — every transcript, tool call, escalation decision, and off-ramp action is logged to Django with timestamps
- **Graceful degradation** — if the AI is uncertain, it defaults to offering a human rather than making a wrong decision with confidence

---

*Project: AI Service — Contestoga Showcase*
*Stack: FastAPI · Django · OpenAI Agents SDK · Twilio · PostgreSQL*
